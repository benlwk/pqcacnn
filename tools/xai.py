"""Interpretability of trained P6 models (saved by exp2.py --save).

1. Patch relevance: the head (average pooling, BatchNorm in eval mode, FC) is affine in the 16 patch outputs, so the
   Shapley value of patch p for a class logit is exact and equals the logit change when p's six Pauli features are
   replaced by their training-set mean at that position (baseline). Faithfulness: accuracy after deleting the k most
   relevant patches (replaced by the baseline) vs k random patches.
2. Head feature importance: the same exact attribution for the 24 pooled features, grouped by qubit and Pauli axis.
3. Channel attention: channel-qubit populations (|c1 c0>) of every patch immediately before and after the QCAL.
4. Block Shapley values (players CCL1, Pool, QCAL, CCL2; a removed block is skipped). Value = test accuracy with the
   trained head (frozen) and with a head re-fitted on the features of that coalition (as in qfollow.py).

usage: python xai.py --ckpt ../ckpt/qG_p6_rgb_s0.pt --cs rgb --spec ../datasets/proto/gtsrb_q.csv --seed 0 --tag ... --out ../logs/xai.jsonl
"""
import argparse, itertools, json, math, os, time
import numpy as np, torch, torch.nn.functional as TF
import pennylane as qml
import exp2, qfollow
import qcacnn as Q
from pqcacnn import PQCACNN, encode_patches, split, readout, SHAPES6, W6, DEV6

BLOCKS = ("CCL1", "Pool", "QCAL", "CCL2")
FEATS = [f"{q}-{p}" for q in ("q3", "q1") for p in ("X", "Y", "Z")]

def make_qnode(mask=(1, 1, 1, 1), probe=None):
    """P6 circuit with blocks switched on/off; probe='pre'/'post' returns channel-qubit probabilities around the QCAL."""
    def f(state, w):
        ccl1, pool, cb, tr, ccl2 = split(w, SHAPES6)
        qml.StatePrep(state, wires=W6, validate_norm=False)
        if mask[0]: Q.controlled_conv_layer(ccl1, ["q3", "q2", "q1", "q0"])
        if mask[1]: Q.transfer(pool[0], "q2", "q3"); Q.transfer(pool[1], "q2", "q1"); Q.transfer(pool[2], "q0", "q1")
        if probe == "pre": return qml.probs(wires=[Q.QC1, Q.QC0])
        if mask[2]: Q.qcal(cb, tr, "q2", "q0")
        if probe == "post": return qml.probs(wires=[Q.QC1, Q.QC0])
        if mask[3]: Q.controlled_conv_layer(ccl2, ["q3", "q1"])
        return readout(["q3", "q1"])
    return qml.QNode(f, DEV6, interface="torch", diff_method="backprop")

@torch.no_grad()
def patch_feats(model, qnode, X, dev):
    """(N, 16, 6) Pauli features per patch."""
    w = model.qw.double(); out = []
    for i in range(0, len(X), 256):
        xb = X[i:i + 256].to(dev)
        f = torch.stack(list(qnode(encode_patches(xb, 4, model.enc), w)), dim=-1).float()
        out.append(f.reshape(xb.shape[0], 16, 6))
    return torch.cat(out)

@torch.no_grad()
def chan_probs(model, qnode, X, dev):
    w = model.qw.double(); out = []
    for i in range(0, len(X), 256):
        xb = X[i:i + 256].to(dev)
        out.append(qnode(encode_patches(xb, 4, model.enc), w).float().reshape(xb.shape[0], 16, 4))
    return torch.cat(out)

def pool(P):                                          # (N, 16, 6) patch features -> (N, 24), same as the model
    return TF.adaptive_avg_pool2d(P.reshape(-1, 4, 4, 6).permute(0, 3, 1, 2), 2).flatten(1)

@torch.no_grad()
def logits(model, P):
    return model.fc(model.bn(pool(P)))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("--ckpt", "--cs", "--spec", "--tag", "--out"): ap.add_argument(k, required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); torch.manual_seed(a.seed); dev = torch.device("cuda"); t0 = time.time()
    Xtr, ytr, te_crops, yte, _ = exp2.load(a.spec, 8, True, a.cs)
    Xte = exp2.to_tensor(te_crops, a.cs); ytr, yte = ytr.to(dev), yte.to(dev); C = Xtr.shape[1]
    model = PQCACNN("P6", "l2"); model.load_state_dict(torch.load(a.ckpt, map_location="cpu")); model = model.to(dev).eval()
    full = make_qnode()
    Ptr, Pte = patch_feats(model, full, Xtr, dev), patch_feats(model, full, Xte, dev)
    base = Ptr.mean(0, keepdim=True)                                   # (1, 16, 6) baseline per patch position
    L = logits(model, Pte); pred = L.argmax(1); N = len(yte)
    acc = 100 * float((pred == yte).float().mean())
    res = dict(tag=a.tag, cs=a.cs, spec=os.path.basename(a.spec), seed=a.seed, acc=round(acc, 2))

    # 1. exact patch Shapley values for the predicted-class logit (affine head -> leave-one-out = Shapley)
    phi = torch.zeros(N, 16, device=dev)
    for p in range(16):
        Q_ = Pte.clone(); Q_[:, p] = base[:, p]
        phi[:, p] = (L - logits(model, Q_)).gather(1, pred[:, None])[:, 0]
    tot = (L - logits(model, base.expand(N, -1, -1))).gather(1, pred[:, None])[:, 0]
    res["shapley_efficiency_err"] = float((phi.sum(1) - tot).abs().max())
    order = phi.argsort(1, descending=True); g = torch.Generator(device="cpu").manual_seed(a.seed)
    dele = {"top": [], "random": []}
    for k in range(17):
        for kind in ("top", "random"):
            accs = []
            for r in range(1 if kind == "top" else 5):
                idx = order[:, :k] if kind == "top" else torch.rand(N, 16, generator=g).argsort(1)[:, :k].to(dev)
                Q_ = Pte.clone()
                if k: Q_.scatter_(1, idx[:, :, None].expand(-1, -1, 6), base.expand(N, -1, -1).gather(1, idx[:, :, None].expand(-1, -1, 6)))
                accs.append(100 * float((logits(model, Q_).argmax(1) == yte).float().mean()))
            dele[kind].append(round(float(np.mean(accs)), 2))
    res["deletion"] = dele

    # 2. exact importance of the 24 pooled features (6 Pauli features x 4 cells), grouped by qubit/axis
    G, Gb = pool(Pte), pool(base)
    imp = torch.zeros(N, 24, device=dev)
    for j in range(24):
        H = G.clone(); H[:, j] = Gb[0, j]
        imp[:, j] = (L - model.fc(model.bn(H))).gather(1, pred[:, None])[:, 0].abs()
    res["feature_importance"] = {FEATS[k]: round(float(imp[:, k * 4:(k + 1) * 4].sum(1).mean()), 4) for k in range(6)}

    # 3. channel-qubit populations around the QCAL
    pre, post = chan_probs(model, make_qnode(probe="pre"), Xte, dev), chan_probs(model, make_qnode(probe="post"), Xte, dev)
    res["channel_pre"] = [[round(float(v), 4) for v in pre[yte == c].mean((0, 1))] for c in range(8)]
    res["channel_post"] = [[round(float(v), 4) for v in post[yte == c].mean((0, 1))] for c in range(8)]

    # 4. block Shapley values (4 players, all 16 coalitions)
    val = {"frozen": {}, "refit": {}}
    for mask in itertools.product((0, 1), repeat=4):
        qn = make_qnode(mask)
        Ftr, Fte = (Ptr, Pte) if all(mask) else (patch_feats(model, qn, Xtr, dev), patch_feats(model, qn, Xte, dev))
        Ftr, Fte = pool(Ftr), pool(Fte)
        key = "".join(map(str, mask))
        val["frozen"][key] = qfollow.score(model.bn, model.fc, Fte, yte, 8)[0]
        bn, fc = qfollow.refit(model.bn, model.fc, Ftr, ytr, a.seed)
        val["refit"][key] = qfollow.score(bn, fc, Fte, yte, 8)[0]
    shap = {}
    for kind in ("frozen", "refit"):
        shap[kind] = {}
        for i, b in enumerate(BLOCKS):
            s = 0.0
            for mask in itertools.product((0, 1), repeat=4):
                if mask[i]: continue
                k = sum(mask); w = math.factorial(k) * math.factorial(3 - k) / math.factorial(4)
                on = list(mask); on[i] = 1
                s += w * (val[kind]["".join(map(str, on))] - val[kind]["".join(map(str, mask))])
            shap[kind][b] = round(s, 2)
    res["block_values"] = val; res["block_shapley"] = shap

    os.makedirs("../xai", exist_ok=True)
    Xrgb = exp2.to_tensor(te_crops, "rgb")
    np.savez_compressed(f"../xai/{a.tag}.npz", phi=phi.cpu().numpy().astype(np.float32), y=yte.cpu().numpy(), pred=pred.cpu().numpy(),
                        pre=pre.cpu().numpy().astype(np.float16), post=post.cpu().numpy().astype(np.float16),
                        img=(Xrgb.numpy() * 255).round().astype(np.uint8))
    with open(a.out, "a") as fh:
        fh.write(json.dumps(res) + "\n")
    print(f"[xai] {a.tag} acc {acc:.2f} | efficiency err {res['shapley_efficiency_err']:.2e} | deletion top {dele['top'][:5]} random {dele['random'][:5]} "
          f"| block Shapley {shap} | {time.time() - t0:.0f}s", flush=True)

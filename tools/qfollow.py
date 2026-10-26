"""Follow-up on the low-precision study, using the P6 checkpoints saved by exp2.py --quant-eval --save.

For each angle precision b, evaluate (i) the original head on features of the angle-quantized circuit and
(ii) a head re-fit: BatchNorm + FC only, re-trained on training-set features of the quantized circuit with the
QCACNN protocol (Adam 0.01, batch 256, 100 epochs). The circuit is never re-trained. 'ref' re-fits the head at
full precision as a control. 'dep{b}': angles, uplink and downlink at b bits, re-fitted head deployed with int8 weights.

usage: python qfollow.py --ckpt ../ckpt/qG_p6_rgb_s0.pt --cs rgb --spec ../datasets/proto/gtsrb_q.csv --seed 0 --tag ... --out ...
"""
import argparse, copy, json, math, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as TF
from sklearn.metrics import f1_score
import exp2
from pqcacnn import PQCACNN, encode_patches

SETTINGS = [("ref", {})] + [(f"ang{b}", dict(ang=b)) for b in (4, 6, 8, 10, 12)] + \
           [(f"dep{b}", dict(ang=b, up=b, down=b, head8=True)) for b in (6, 8)]

@torch.no_grad()
def features(model, cs, C, X, dev, ang=None, up=None, down=None):
    """(N, 24) pooled Pauli features, before BatchNorm, under the given quantization."""
    w = model.qw.double()
    if ang: w = exp2.q_angles(w, ang)
    m = C * model.k * model.k; out = []
    for i in range(0, len(X), 256):
        xb = X[i:i + 256].to(dev); B = xb.shape[0]
        a = encode_patches(xb, model.k, model.enc)
        if up:
            v = exp2.q_unsigned(a[:, :m], up) if cs in exp2.NONNEG else exp2.q_signed(a[:, :m], up)
            n = v.norm(dim=1, keepdim=True)
            v = torch.where(n > 1e-12, v / n.clamp_min(1e-12), torch.full_like(v, 1 / math.sqrt(m)))
            a = torch.cat([v, a[:, m:]], dim=1)
        f = torch.stack(list(model.qnode(a, w)), dim=-1).float()
        if down: f = exp2.q_signed(f, down)
        out.append(TF.adaptive_avg_pool2d(f.reshape(B, 4, 4, 6).permute(0, 3, 1, 2), 2).flatten(1))
    return torch.cat(out)

def refit(bn, fc, F, y, seed, epochs=100, batch=256, lr=0.01):
    bn, fc = copy.deepcopy(bn), copy.deepcopy(fc); head = nn.Sequential(bn, fc).train()
    opt = torch.optim.Adam(head.parameters(), lr=lr); crit = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(y), generator=g).to(F.device)
        for i in range(0, len(y), batch):
            idx = perm[i:i + batch]; opt.zero_grad(); crit(head(F[idx]), y[idx]).backward(); opt.step()
    return bn.eval(), fc.eval()

@torch.no_grad()
def score(bn, fc, F, y, n_classes, head8=False):
    if head8:
        s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        W = fc.weight * s; b = fc.bias + fc.weight @ (bn.bias - bn.running_mean * s)
        step = W.abs().max() / 127; logits = F @ (torch.round(W / step) * step).t() + b
    else:
        logits = fc(bn(F))
    pred = logits.argmax(1).cpu().numpy(); yy = y.cpu().numpy()
    return [round(100 * float((pred == yy).mean()), 2),
            round(100 * f1_score(yy, pred, labels=list(range(n_classes)), average="macro", zero_division=0), 2)]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("--ckpt", "--cs", "--spec", "--tag", "--out"): ap.add_argument(k, required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); torch.manual_seed(a.seed); dev = torch.device("cuda"); t0 = time.time()
    Xtr, ytr, te_crops, yte, _ = exp2.load(a.spec, 8, True, a.cs)      # same balanced training subset as in training
    Xte = exp2.to_tensor(te_crops, a.cs); ytr, yte = ytr.to(dev), yte.to(dev); C = Xtr.shape[1]
    model = PQCACNN("P6", "l2"); model.load_state_dict(torch.load(a.ckpt, map_location="cpu")); model = model.to(dev).eval()
    res = dict(tag=a.tag, cs=a.cs, spec=a.spec, seed=a.seed, frozen={}, refit={})
    for name, kw in SETTINGS:
        h8 = kw.pop("head8", False)
        Ftr, Fte = features(model, a.cs, C, Xtr, dev, **kw), features(model, a.cs, C, Xte, dev, **kw)
        res["frozen"][name] = score(model.bn, model.fc, Fte, yte, 8, h8)
        bn, fc = refit(model.bn, model.fc, Ftr, ytr, a.seed)
        res["refit"][name] = score(bn, fc, Fte, yte, 8, h8)
        print(f"[qfollow] {name:6s} frozen {res['frozen'][name][0]:6.2f}  refit {res['refit'][name][0]:6.2f}  ({time.time() - t0:.0f}s)", flush=True)
    with open(a.out, "a") as fh:
        fh.write(json.dumps(res) + "\n")

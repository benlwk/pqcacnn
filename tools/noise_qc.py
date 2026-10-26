"""QCACNN's single-qubit noise protocol (Qu et al. 2025, Sec. IV-C-4, Table V) applied to trained P6 and QCACNN models.

A noise layer applies one channel -- bit flip (BF), phase flip (PF), depolarizing (DP) or amplitude damping (AD) --
with probability p in {0.01, 0.05, 0.1} to every qubit, once after the encoding and once after the first QCAL.
Circuits are simulated as density matrices (default.mixed). Models are trained without noise (exp2.py --save) and
the channels are applied at inference; QCACNN injected the noise during training instead.

The noisy circuits mirror pqcacnn.p6_ops and qcacnn.circuit_ops; at p = 0 their outputs are checked against the
original QNodes before any noisy evaluation.

usage: python noise_qc.py --model p6|qcacnn --ckpt ../ckpt/x.pt --spec ../datasets/proto/gtsrb_q.csv --tag x --out ../logs/noise_qc.jsonl
"""
import argparse, json, time
import numpy as np, torch, torch.nn.functional as TF
import pennylane as qml
from sklearn.metrics import f1_score
import qcacnn as Q
import pqcacnn as P
import exp2

CHANNELS = {"BF": qml.BitFlip, "PF": qml.PhaseFlip, "DP": qml.DepolarizingChannel, "AD": qml.AmplitudeDamping}
PROBS = [0.01, 0.05, 0.1]

def noise(kind, p, wires):
    if p > 0:
        for w in wires:
            CHANNELS[kind](p, wires=w)

def p6_noisy(state, w, kind, p):                      # mirrors pqcacnn.p6_ops
    ccl1, pool, cb, tr, ccl2 = P.split(w, P.SHAPES6)
    qml.StatePrep(state, wires=P.W6, validate_norm=False)
    noise(kind, p, P.W6)                                                   # after encoding
    Q.controlled_conv_layer(ccl1, ["q3", "q2", "q1", "q0"])
    Q.transfer(pool[0], "q2", "q3"); Q.transfer(pool[1], "q2", "q1"); Q.transfer(pool[2], "q0", "q1")
    Q.qcal(cb, tr, "q2", "q0")
    noise(kind, p, P.W6)                                                   # after the (first and only) QCAL
    Q.controlled_conv_layer(ccl2, ["q3", "q1"])
    return P.readout(["q3", "q1"])

def qcacnn_noisy(state, w, kind, p):                  # mirrors qcacnn.circuit_ops
    ccl1, pool2, cb3, tr3, ccl4, pool5, cb6, tr6, ccl7 = Q.unpack(w)
    qml.StatePrep(state, wires=Q.WIRES, validate_norm=False)
    noise(kind, p, Q.WIRES)                                                # after encoding
    q = {i: f"q{i}" for i in range(8)}
    Q.controlled_conv_layer(ccl1, [q[7], q[6], q[5], q[4], q[3], q[2], q[1], q[0]])
    Q.pooling(pool2, q[5], q[4], q[2], q[1], q[0]); Q.qcal(cb3, tr3, q[4], q[1])
    noise(kind, p, Q.WIRES)                                                # after the first QCAL
    Q.controlled_conv_layer(ccl4, [q[7], q[6], q[5], q[3], q[2], q[0]])
    Q.pooling(pool5, q[7], q[6], q[5], q[3], q[2]); Q.qcal(cb6, tr6, q[6], q[3])
    Q.controlled_conv_layer(ccl7, [q[7], q[5], q[2], q[0]])
    return qml.probs(wires=[q[7], q[5], q[2], q[0]])

def qnode(model_name, dev_name):
    fn, wires = (p6_noisy, P.W6) if model_name == "p6" else (qcacnn_noisy, Q.WIRES)
    return qml.QNode(fn, qml.device(dev_name, wires=wires), interface="torch", diff_method=None)

@torch.no_grad()
def logits(model, model_name, circ, xb, kind, p):
    if model_name == "p6":
        out = circ(P.encode_patches(xb, model.k, model.enc), model.qw.double(), kind, p)
        f = torch.stack(list(out), dim=-1).float()
        f = TF.adaptive_avg_pool2d(f.reshape(xb.shape[0], 4, 4, 6).permute(0, 3, 1, 2), 2).flatten(1)
        return model.fc(model.bn(f))
    return model.fc(circ(Q.encode(xb, model.T, model.amp), model.qw.double(), kind, p).float())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["p6", "qcacnn"]); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--spec", required=True); ap.add_argument("--cs", default="rgb")
    ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=0); ap.add_argument("--limit", type=int, default=0, help="first N test images (timing)")
    ap.add_argument("--channels", default="BF,PF,DP,AD")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); t0 = time.time()
    _, _, te_crops, yte, _ = exp2.load(a.spec, 8, False, a.cs)
    Xte = exp2.to_tensor(te_crops, a.cs)
    if a.limit: Xte, yte = Xte[: a.limit], yte[: a.limit]
    model = exp2.build(a.model, 8, Xte.shape[1]); model.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
    model = model.to(dev).eval()
    bs = a.batch or (256 if a.model == "p6" else 16)
    mixed = qnode(a.model, "default.mixed")

    def run(kind, p):
        preds = [logits(model, a.model, mixed, Xte[i:i + bs].to(dev), kind, p).argmax(1).cpu() for i in range(0, len(yte), bs)]
        pred, y = torch.cat(preds).numpy(), yte.numpy()
        return [round(100 * float((pred == y).mean()), 2), round(100 * f1_score(y, pred, labels=list(range(8)), average="macro", zero_division=0), 2)]

    with torch.no_grad():                                                  # mirror check against the training QNode
        xb = Xte[:4].to(dev)
        a_ref = model(xb); a_mix = logits(model, a.model, mixed, xb, "BF", 0.0)
        err = float((a_ref - a_mix).abs().max())
        print(f"[noise_qc] mirror check: max |logit diff| at p=0 = {err:.2e}", flush=True)
        assert err < 1e-4, "noisy circuit does not reproduce the trained model"
    res = dict(tag=a.tag, model=a.model, cs=a.cs, spec=a.spec.split("/")[-1], ckpt=a.ckpt, n_test=len(yte), noise={})
    tc = time.time(); res["noise"]["clean"] = run("BF", 0.0)
    print(f"[noise_qc] clean {res['noise']['clean']} ({time.time() - tc:.0f}s on {dev}, batch {bs})", flush=True)
    for kind in a.channels.split(","):
        for p in PROBS:
            tc = time.time(); res["noise"][f"{kind}_{p:g}"] = r = run(kind, p)
            print(f"[noise_qc] {kind} p={p:g} acc {r[0]:6.2f} F1 {r[1]:6.2f} ({time.time() - tc:.0f}s)", flush=True)
    res["seconds"] = round(time.time() - t0)
    with open(a.out, "a") as fh:
        fh.write(json.dumps(res) + "\n")
    print(f"[noise_qc] done in {time.time() - t0:.0f}s", flush=True)

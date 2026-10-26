"""Gradient variance (barren-plateau test) of global vs patch-wise channel-aware QCNNs.

mode circuit: Var over random parameters theta ~ U(0, 2pi) and random data inputs x of dC(x, theta)/dtheta_k,
  averaged over all parameters k (and separately over the first CCL), for two readouts:
    z  : <Z> on the first measured qubit (a local observable),
    p0 : probability of the all-zero outcome on the measured qubits (QCACNN's readout).
  Circuits:
    G<f>    : QCACNN's blocks on f feature + 2 channel qubits, globally encoding an s x s RGB image with s = 2^(f/2)
              (f = 4..12, i.e. 4x4 .. 64x64 images, 6..14 qubits): CCL -> [pool + QCAL, removing 2 qubits] -> ... -> CCL
              until 4 feature qubits remain (QCACNN's rule; G8 has QCACNN's 292 parameters but a regular pooling window).
    QCACNN  : the exact re-implemented 10-qubit circuit (16x16 image), with its sqrt-softmax encoding or unit norm.
    P6, P4  : the patch-wise circuits on 4x4 / 2x2 patches of s x s images (circuit size independent of s).
  Inputs are GTSRB test crops resized to s x s; unit-norm amplitude encoding unless stated.
mode loss: squared norm of the training-loss gradient w.r.t. the quantum parameters at initialisation, for the actual
  P6 and QCACNN models (own heads, PyTorch default init), on 256 balanced GTSRB training images, over 20 seeds.

usage: python gradvar.py --mode circuit --circuit G8 --size 16 --M 400 --out ../logs/gradvar.jsonl
       python gradvar.py --mode loss --model p6 --out ../logs/gradvar.jsonl
"""
import argparse, csv, json, math, time
import numpy as np, torch, cv2
import pennylane as qml
import qcacnn as Q
import pqcacnn as P
import exp2

SPEC = "../datasets/proto/gtsrb_q.csv"

def global_plan(f):
    active = [f"q{i}" for i in reversed(range(f))]           # top -> bottom, q0 = most significant pixel bit
    plan, stage = [], 0
    while True:
        plan.append(("ccl", list(active)))
        if len(active) <= 4:
            return plan, active
        win = active[-5:] if stage % 2 == 0 else active[:5]     # 5-qubit pooling window; win[1], win[3] are discarded
        plan.append(("pool", win))
        active = [a for a in active if a not in (win[1], win[3])]; stage += 1

def global_ops(state, w, f):
    plan, _ = global_plan(f)
    qml.StatePrep(state, wires=[Q.QC1, Q.QC0] + [f"q{i}" for i in range(f)], validate_norm=False); i = 0
    for kind, ws in plan:
        if kind == "ccl":
            Q.controlled_conv_layer(w[i:i + 80].reshape(4, 2, 10), ws); i += 80
        else:
            r4, r3, r2, r1, r0 = ws
            Q.pooling(w[i:i + 8].reshape(4, 2), r4, r3, r2, r1, r0)
            Q.qcal(w[i + 8:i + 18], w[i + 18:i + 26].reshape(4, 2), r3, r1); i += 26

def n_global(f):
    return sum(80 if k == "ccl" else 26 for k, _ in global_plan(f)[0])

def images(size, n, seed):
    rows = [r for r in csv.DictReader(open(SPEC)) if r["split"] == "test"]
    rng = np.random.default_rng(seed); pick = [rows[i] for i in rng.choice(len(rows), n, replace=False)]
    arr = [cv2.cvtColor(cv2.resize(exp2.read_crop(r), (size, size), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB) for r in pick]
    return torch.tensor(np.stack(arr), dtype=torch.float64).permute(0, 3, 1, 2) / 255.0

def l2_global(x):
    v = x.reshape(x.shape[0], -1); v = v / v.norm(dim=1, keepdim=True)
    return torch.cat([v, torch.zeros(v.shape[0], v.shape[1] // 3, dtype=v.dtype)], dim=1)   # channel |11> empty

def setup(name, size, enc, n_inputs, seed):
    """-> ops(state, w), n_params, measured wires (first = Z qubit), input states (n_inputs, 2^n), first-CCL param count"""
    x = images(size, max(n_inputs, 64), seed)
    if name.startswith("G"):
        f = int(name[1:]); assert size == 2 ** (f // 2), "G<f> encodes an image with 2^f pixels"
        meas = global_plan(f)[1]
        return (lambda s, w: global_ops(s, w, f)), n_global(f), meas, l2_global(x)[:n_inputs], 80
    if name == "QCACNN":
        assert size == 16
        st = Q.encode(x.float(), 0.1, "sqrt") if enc == "sqrtsoftmax" else l2_global(x)
        return (lambda s, w: Q.circuit_ops(s, w)), Q.N_QPARAMS, ["q7", "q5", "q2", "q0"], st[:n_inputs].double(), 80
    k = 4 if name == "P6" else 2
    A = P.encode_patches(x, k, "l2")
    A = A[A[:, : 3 * k * k].norm(dim=1) > 0]                              # skip all-black patches
    A = A[torch.randperm(len(A), generator=torch.Generator().manual_seed(seed))[:n_inputs]]
    if name == "P6":
        return P.p6_ops, 144, ["q3", "q1"], A, 80
    return P.p4_ops, 120, ["q1", "q0"], A, 40

def circuit_mode(a):
    ops, npar, meas, X, n_first = setup(a.circuit, a.size, a.enc, a.M, a.seed)
    wires = qml.tape.make_qscript(ops)(X[:1], torch.zeros(npar, dtype=torch.float64)).wires
    dev = qml.device("default.qubit", wires=wires)
    rng = torch.Generator().manual_seed(a.seed); G = {"z": [], "p0": []}; t0 = time.time()
    for m in range(a.M):                                      # one (theta, x) pair per sample
        w = (torch.rand(npar, generator=rng, dtype=torch.float64) * 2 * math.pi).requires_grad_(True)
        tape = qml.tape.make_qscript(ops)(X[m:m + 1], w)
        tape = qml.tape.QuantumScript(tape.operations, [qml.expval(qml.PauliZ(meas[0])), qml.probs(wires=meas)])
        z, pr = qml.execute([tape], dev, diff_method="backprop")[0]
        gz, = torch.autograd.grad(z.reshape(-1)[0], w, retain_graph=True)
        gp, = torch.autograd.grad(pr.reshape(-1)[0], w)
        G["z"].append(gz.numpy()); G["p0"].append(gp.numpy())
        if m == 0: print(f"[gradvar] {a.circuit} first sample {time.time() - t0:.1f}s", flush=True)
    res = dict(mode="circuit", circuit=a.circuit, size=a.size, enc=a.enc, qubits=len(wires), params=npar, M=a.M,
               seconds=round(time.time() - t0))
    for obs, g in G.items():
        g = np.stack(g); v = g.var(axis=0)
        res[f"var_{obs}"] = float(v.mean()); res[f"var_{obs}_first"] = float(v[:n_first].mean())
        res[f"var_{obs}_median"] = float(np.median(v))
    return res

def feat_mode(a):
    """Feature concentration: for T random parameter sets, the variance across D data inputs of the readout features
    (<Z> on the first measured qubit, and the outcome probabilities), averaged over parameter sets."""
    ops, npar, meas, X, _ = setup(a.circuit, a.size, a.enc, a.D, a.seed)
    wires = qml.tape.make_qscript(ops)(X[:1], torch.zeros(npar, dtype=torch.float64)).wires
    dev = qml.device("default.qubit", wires=wires)
    rng = torch.Generator().manual_seed(a.seed); vz, vp, t0 = [], [], time.time()
    for _ in range(a.T):
        w = torch.rand(npar, generator=rng, dtype=torch.float64) * 2 * math.pi
        tape = qml.tape.make_qscript(ops)(X, w)
        tape = qml.tape.QuantumScript(tape.operations, [qml.expval(qml.PauliZ(meas[0])), qml.probs(wires=meas)])
        z, pr = qml.execute([tape], dev)[0]
        vz.append(float(torch.as_tensor(z).var())); vp.append(float(torch.as_tensor(pr).var(0).mean()))
    return dict(mode="feat", circuit=a.circuit, size=a.size, enc=a.enc, qubits=len(wires), D=len(X), T=a.T,
                featvar_z=float(np.mean(vz)), featvar_p=float(np.mean(vp)), seconds=round(time.time() - t0))

def loss_mode(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xtr, ytr, _, _, _ = exp2.load(SPEC, 8, True, "rgb")
    idx = torch.randperm(len(ytr), generator=torch.Generator().manual_seed(0))[:256]
    xb, yb = Xtr[idx].to(dev), ytr[idx].to(dev); out = []
    for s in range(a.seeds):
        torch.manual_seed(s); m = exp2.build(a.model, 8).to(dev).train()
        loss = torch.nn.functional.cross_entropy(m(xb), yb)
        g, = torch.autograd.grad(loss, m.qw)
        out.append(float((g ** 2).mean()))
    return dict(mode="loss", model=a.model, seeds=a.seeds, grad2_mean=float(np.mean(out)), grad2_std=float(np.std(out)),
                grad2_all=[round(v, 8) for v in out])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["circuit", "loss", "feat"])
    ap.add_argument("--D", type=int, default=256); ap.add_argument("--T", type=int, default=50)
    ap.add_argument("--circuit"); ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--enc", default="l2", choices=["l2", "sqrtsoftmax"])
    ap.add_argument("--M", type=int, default=400); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", choices=["p6", "qcacnn", "qcacnn_bn", "p6_nobn"]); ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"circuit": circuit_mode, "loss": loss_mode, "feat": feat_mode}[a.mode](a)
    print(json.dumps(res), flush=True)
    with open(a.out, "a") as fh:
        fh.write(json.dumps(res) + "\n")

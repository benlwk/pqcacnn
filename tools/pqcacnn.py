"""Patch-wise QCACNN prototype (fewer qubits, same parameter budget as QCACNN's 428).

Idea 1: amplitude-encode small colour patches instead of the whole image, keeping QCACNN's channel qubits,
        channel-controlled convolution, symmetric pooling and channel-attention layer (building blocks from qcacnn.py).
Idea 4: read out Pauli X/Y/Z expectation values instead of a full outcome distribution (smaller classical head).

  P6: 4x4x3 patch -> 6 qubits (2 channel + 4 feature), 16 patches/image, 144 quantum params, 392 total
  P4: 2x2x3 patch -> 4 qubits (2 channel + 2 feature), 64 patches/image, 40*layers quantum params, 368 total (3 layers)

Head: per-patch readouts form a (6, g, g) map -> average-pool to 2x2 -> BatchNorm -> Linear(24, 8).
Training protocol identical to qcacnn.py (QCACNN Sec. IV-A): Adam 0.01, batch 256, 100 iterations, final epoch reported.

usage: python pqcacnn.py --arch P6|P4 --enc sqrtsoftmax|l2 --spec <split.csv> --seed 0 [--check]
"""
import argparse, math, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import pennylane as qml
from sklearn.metrics import f1_score
import qcacnn as Q                                   # conv_block / transfer / controlled conv / QCAL / data loader / CNN

PAULIS = (qml.PauliX, qml.PauliY, qml.PauliZ)

def readout(wires):
    return [qml.expval(P(w)) for w in wires for P in PAULIS]

# ---- P6: 6 qubits [qc1, qc0, q0..q3]; q0 = most significant pixel-index bit
W6 = [Q.QC1, Q.QC0, "q0", "q1", "q2", "q3"]
SHAPES6 = [(4, 2, 10), (3, 2), (10,), (4, 2), (4, 1, 10)]       # ccl1 80, pool 6, qcal 10+8, ccl2 40 = 144

def p6_ops(state, w):
    qml.StatePrep(state, wires=W6, validate_norm=False)
    p6_body(w)
    return readout(["q3", "q1"])

def p6_body(w):                                      # the trainable circuit after state preparation
    ccl1, pool, cb, tr, ccl2 = split(w, SHAPES6)
    Q.controlled_conv_layer(ccl1, ["q3", "q2", "q1", "q0"])
    Q.transfer(pool[0], "q2", "q3"); Q.transfer(pool[1], "q2", "q1"); Q.transfer(pool[2], "q0", "q1")   # double pooling of q2; q0 -> q1
    Q.qcal(cb, tr, "q2", "q0")                                                                         # discarded q2, q0 -> channel qubits
    Q.controlled_conv_layer(ccl2, ["q3", "q1"])

# ---- fast exact P6 readout. All patches share one circuit U(w), so each readout is <psi| U^dag O U |psi>; with real
# amplitudes psi this equals psi^T Re(U^dag O U) psi. U is built once per forward pass (64 x 64) instead of simulating
# every patch; gradients flow through qml.matrix. Equal to P6_QNODE to machine precision (see --check-fast).
_OBS6 = None

def p6_heisenberg(w):
    global _OBS6
    if _OBS6 is None:
        _OBS6 = torch.stack([torch.as_tensor(qml.matrix(P(q), wire_order=W6), dtype=torch.complex128)
                             for q in ("q3", "q1") for P in PAULIS])
    U = qml.matrix(p6_body, wire_order=W6)(w.cpu())
    return (U.conj().T @ _OBS6 @ U).real                                  # (6, 64, 64)

def p6_fast(states, w):
    M = p6_heisenberg(w).to(states.device)
    return torch.einsum("ni,jik,nk->nj", states, M, states)              # (N, 6), same order as readout()

# ---- P4: 4 qubits [qc1, qc0, q0, q1]
W4 = [Q.QC1, Q.QC0, "q0", "q1"]

def p4_ops(state, w, layers=3):
    qml.StatePrep(state, wires=W4, validate_norm=False)
    for t in split(w, [(4, 1, 10)] * layers):
        Q.controlled_conv_layer(t, ["q1", "q0"])
    return readout(["q1", "q0"])

def split(w, shapes):
    out, i = [], 0
    for s in shapes:
        n = int(np.prod(s)); out.append(w[i:i + n].reshape(s)); i += n
    assert i == len(w), (i, len(w))
    return out

DEV6, DEV4 = qml.device("default.qubit", wires=W6), qml.device("default.qubit", wires=W4)
P6_QNODE = qml.QNode(p6_ops, DEV6, interface="torch", diff_method="backprop")
P4_QNODE = qml.QNode(p4_ops, DEV4, interface="torch", diff_method="backprop")

def encode_patches(x, k, enc, T=0.1):
    """(B,C,16,16), C in {3,4} -> (B*L, 4*k*k) amplitude vectors; index = channel*k*k + row*k + col.
    With C=3 the channel state |11> is left empty; C=4 fills it (no extra qubits)."""
    B, C = x.shape[0], x.shape[1]
    v = F.unfold(x.double(), kernel_size=k, stride=k).transpose(1, 2).reshape(-1, C * k * k)
    if enc == "sqrtsoftmax":
        a = torch.softmax(v / T, dim=1).sqrt()
    else:                                            # l2: plain amplitude encoding; all-black patch -> uniform
        n = v.norm(dim=1, keepdim=True)
        a = torch.where(n > 1e-8, v / n.clamp_min(1e-8), torch.full_like(v, 1 / math.sqrt(v.shape[1])))
    return torch.cat([a, torch.zeros(a.shape[0], (4 - C) * k * k, dtype=a.dtype, device=a.device)], dim=1)

class PQCACNN(nn.Module):
    def __init__(self, arch="P6", enc="sqrtsoftmax", layers=3, fast=False, pool=2):
        super().__init__()
        self.arch, self.enc, self.fast, self.pool = arch, enc, fast and arch == "P6", pool
        self.k = 4 if arch == "P6" else 2
        self.nq = sum(int(np.prod(s)) for s in SHAPES6) if arch == "P6" else 40 * layers
        self.qnode = P6_QNODE if arch == "P6" else P4_QNODE
        self.qw = nn.Parameter(torch.rand(self.nq) * 2 * math.pi)   # U(0, 2pi), as for QCACNN
        self.bn = nn.BatchNorm1d(6 * pool * pool)
        self.fc = nn.Linear(6 * pool * pool, 8)
    def patch_features(self, x):
        """(B, C, H, W) -> (B, 6, H/k, W/k) map of Pauli readouts, one per non-overlapping k x k patch."""
        B, g = x.shape[0], x.shape[-1] // self.k
        a = encode_patches(x, self.k, self.enc)
        f = p6_fast(a, self.qw.double()) if self.fast else torch.stack(list(self.qnode(a, self.qw.double())), dim=-1)
        return f.float().reshape(B, g, g, 6).permute(0, 3, 1, 2)
    def forward(self, x):
        f = F.adaptive_avg_pool2d(self.patch_features(x), self.pool).flatten(1)   # (B, 6 * pool^2)
        return self.fc(self.bn(f))

# ---- resource accounting (per circuit, after full decomposition incl. state preparation)
def resources():
    gate_set = {"CNOT", "RX", "RY", "RZ", "PhaseShift", "GlobalPhase"}
    rng = np.random.default_rng(0)
    def count(qfunc, n_amp, n_par, label, circuits_per_image):
        st = rng.random(n_amp); st /= np.linalg.norm(st)
        w = rng.random(n_par) * 2 * np.pi
        tape = qml.tape.make_qscript(qfunc)(st, w)
        logical = [op for op in tape.operations if not isinstance(op, qml.StatePrep)]
        try:
            (dt,), _ = qml.transforms.decompose(tape, gate_set=gate_set)
            cx = sum(op.name == "CNOT" for op in dt.operations)
            sp = qml.tape.make_qscript(lambda s: qml.StatePrep(s, wires=range(int(np.log2(n_amp)))))(st)
            (dsp,), _ = qml.transforms.decompose(sp, gate_set=gate_set)
            cx_sp = sum(op.name == "CNOT" for op in dsp.operations)
            depth = dt.specs["resources"].depth
        except Exception as e:                       # decomposition support varies across PennyLane versions
            cx = cx_sp = depth = f"n/a ({type(e).__name__})"
        print(f"[resources] {label:7s} qubits {len(tape.wires):2d} | logical gates {len(logical):4d} | "
              f"CNOTs after decomposition {cx} (state prep alone {cx_sp}) | depth {depth} | circuits/image {circuits_per_image}")
    count(lambda s, w: Q.circuit_ops(s, torch.tensor(w)), 1024, Q.N_QPARAMS, "QCACNN", 1)
    count(p6_ops, 64, 144, "P6", 16)
    count(p4_ops, 16, 120, "P4", 64)
    for arch in ("P6", "P4"):
        m = PQCACNN(arch); print(f"[resources] {arch} total params {sum(p.numel() for p in m.parameters())} (quantum {m.nq})")

def check_fast(n=64, seed=0):
    """Fast Heisenberg readout vs the PennyLane QNode: values and gradients on random unit-norm colour patches."""
    g = torch.Generator().manual_seed(seed)
    a = encode_patches(torch.rand(n // 16, 4, 16, 16, generator=g, dtype=torch.float64) - 0.3, 4, "l2")
    w1 = (torch.rand(144, generator=g, dtype=torch.float64) * 2 * math.pi).requires_grad_(True)
    w2 = w1.detach().clone().requires_grad_(True)
    ref = torch.stack(list(P6_QNODE(a, w1)), dim=-1); fast = p6_fast(a, w2)
    (gr,) = torch.autograd.grad(ref.sum() + (ref ** 2).sum(), w1); (gf,) = torch.autograd.grad(fast.sum() + (fast ** 2).sum(), w2)
    print(f"[check-fast] {len(a)} patches | max |value diff| {float((ref - fast).abs().max()):.2e} | "
          f"max |grad diff| {float((gr - gf).abs().max()):.2e} (grad scale {float(gr.abs().max()):.2e})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="P6", choices=["P6", "P4"]); ap.add_argument("--enc", default="sqrtsoftmax", choices=["sqrtsoftmax", "l2"])
    ap.add_argument("--layers", type=int, default=3); ap.add_argument("--spec"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.01); ap.add_argument("--check", action="store_true")
    ap.add_argument("--check-fast", action="store_true")
    a = ap.parse_args()
    if a.check_fast:
        check_fast(); raise SystemExit
    if a.check:
        resources(); raise SystemExit
    torch.manual_seed(a.seed); dev = torch.device("cuda")
    t0 = time.time(); Xtr, ytr, Xte, yte = (t.to(dev) for t in Q.load(a.spec))
    model = PQCACNN(a.arch, a.enc, a.layers).to(dev)
    print(f"[pqcacnn] {a.arch} enc={a.enc} spec={os.path.basename(a.spec)} seed={a.seed} | train {tuple(Xtr.shape)} test {tuple(Xte.shape)} "
          f"| params {sum(p.numel() for p in model.parameters())} (quantum {model.nq}) | data {time.time() - t0:.0f}s", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr); crit = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(a.seed)
    for ep in range(1, a.epochs + 1):
        te = time.time(); model.train(); perm = torch.randperm(len(ytr), generator=g).to(dev); tl = 0.0
        for i in range(0, len(ytr), a.batch):
            idx = perm[i:i + a.batch]; opt.zero_grad()
            loss = crit(model(Xtr[idx]), ytr[idx]); loss.backward(); opt.step(); tl += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            pred = torch.cat([model(Xte[i:i + 512]).argmax(1) for i in range(0, len(yte), 512)]).cpu().numpy()
        y = yte.cpu().numpy(); acc = 100 * float((pred == y).mean())
        f1 = 100 * f1_score(y, pred, labels=list(range(8)), average="macro", zero_division=0)
        print(f"Epoch {ep:03d}/{a.epochs} | Train Loss: {tl / len(ytr):.4f} | Test Acc: {acc:.2f}% | Test F1: {f1:.2f}% | {time.time() - te:.1f}s", flush=True)
    print(f"[pqcacnn] done in {time.time() - t0:.0f}s", flush=True)

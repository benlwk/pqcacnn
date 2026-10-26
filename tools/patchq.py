"""P8: patch-wise channel-aware QCNN on 8x8 colour patches (2 channel + 6 feature qubits).

Circuit = QCACNN's blocks on 6 feature qubits (gradvar.global_plan(6): CCL -> pooling + QCAL -> CCL, 186 parameters),
applied to every non-overlapping 8x8 patch; Pauli X/Y/Z on the 4 remaining feature qubits -> 12 features per patch.
Readout is exact and fast in the Heisenberg picture: one 256 x 256 unitary per forward pass, features psi^T Re(U^dag O U) psi.

usage: python patchq.py --check      (fast readout vs PennyLane QNode, values and gradients)
"""
import math, sys
import torch, torch.nn as nn, torch.nn.functional as F
import pennylane as qml
import qcacnn as Q
import gradvar
from pqcacnn import encode_patches, PAULIS

def body(w, f):                                         # gradvar.global_ops without the state preparation
    i = 0
    for kind, ws in gradvar.global_plan(f)[0]:
        if kind == "ccl":
            Q.controlled_conv_layer(w[i:i + 80].reshape(4, 2, 10), ws); i += 80
        else:
            r4, r3, r2, r1, r0 = ws
            Q.pooling(w[i:i + 8].reshape(4, 2), r4, r3, r2, r1, r0)
            Q.qcal(w[i + 8:i + 18], w[i + 18:i + 26].reshape(4, 2), r3, r1); i += 26

class P8(nn.Module):
    def __init__(self, n_classes, pool=2, f=6):
        super().__init__()
        self.f, self.k, self.pool = f, 2 ** (f // 2), pool
        self.wires = [Q.QC1, Q.QC0] + [f"q{i}" for i in range(f)]
        self.meas = gradvar.global_plan(f)[1]
        self.register_buffer("obs", torch.stack([torch.as_tensor(qml.matrix(P(q), wire_order=self.wires), dtype=torch.complex128)
                                                 for q in self.meas for P in PAULIS]), persistent=False)
        self.nf = 3 * len(self.meas)
        self.qw = nn.Parameter(torch.rand(gradvar.n_global(f)) * 2 * math.pi)
        self.bn = nn.BatchNorm1d(self.nf * pool * pool)
        self.fc = nn.Linear(self.nf * pool * pool, n_classes)
    def heisenberg(self):
        U = qml.matrix(lambda w: body(w, self.f), wire_order=self.wires)(self.qw.double().cpu())
        return (U.conj().T @ self.obs.cpu() @ U).real
    def patch_features(self, x):
        B, g = x.shape[0], x.shape[-1] // self.k
        a = encode_patches(x, self.k, "l2")
        M = self.heisenberg().to(a.device)
        f = torch.einsum("ni,jik,nk->nj", a, M, a)
        return f.float().reshape(B, g, g, self.nf).permute(0, 3, 1, 2)
    def forward(self, x):
        return self.fc(self.bn(F.adaptive_avg_pool2d(self.patch_features(x), self.pool).flatten(1)))

def check(seed=0):
    torch.manual_seed(seed); m = P8(8)
    x = torch.rand(2, 3, 16, 16, dtype=torch.float64)
    a = encode_patches(x, 8, "l2")
    def circ(s, w):
        qml.StatePrep(s, wires=m.wires, validate_norm=False); body(w, m.f)
        return [qml.expval(P(q)) for q in m.meas for P in PAULIS]
    qn = qml.QNode(circ, qml.device("default.qubit", wires=m.wires), interface="torch", diff_method="backprop")
    w1 = m.qw.detach().double().clone().requires_grad_(True)
    ref = torch.stack(list(qn(a, w1)), dim=-1)
    fast = torch.einsum("ni,jik,nk->nj", a, m.heisenberg(), a)
    (gr,) = torch.autograd.grad(ref.sum() + (ref ** 2).sum(), w1)
    (gf,) = torch.autograd.grad(fast.sum() + (fast ** 2).sum(), m.qw)
    print(f"[P8 check] {len(a)} patches, {m.nf} features, {sum(p.numel() for p in m.parameters())} params | "
          f"max |value diff| {float((ref - fast).abs().max()):.2e} | max |grad diff| {float((gr - gf.double()).abs().max()):.2e}")

if __name__ == "__main__" and "--check" in sys.argv:
    check()

# Classical simulation cost per image (dense state-vector): an op on k wires of an n-qubit state costs
# about 2^n * 2^k complex multiply-adds = 8 * 2^n * 2^k real FLOPs (state preparation counted as 2^n * 8).
import numpy as np, torch, pennylane as qml
import qcacnn as Q, pqcacnn as P
def cost(qfunc, n_amp, n_par, nq):
    st = np.random.rand(n_amp); st /= np.linalg.norm(st)
    tape = qml.tape.make_qscript(qfunc)(st, np.random.rand(n_par))
    return sum(8 * 2 ** nq * (2 ** len(op.wires) if not isinstance(op, qml.StatePrep) else 1) for op in tape.operations)
c_q = cost(lambda s, w: Q.circuit_ops(s, torch.tensor(w)), 1024, 292, 10)
c_p6 = cost(P.p6_ops, 64, 144, 6)
print(f"[flops] QCACNN {c_q:.3g} per image (1 circuit) | P6 {c_p6:.3g} per circuit, {16 * c_p6:.3g} per image (16 circuits) | CNN ~{2 * 30080:.3g}")

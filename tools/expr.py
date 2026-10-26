"""Expressibility and entangling capability (Sim et al., Adv. Quantum Technol. 2019) of the P6, P4 and QCACNN circuits.

Protocol as in QCACNN Sec. IV-C-3: fixed input |0...0>, M random parameter sets drawn U(0, 2pi) (the training
initialisation), fidelities of all M(M-1)/2 pairs of output states, 75-bin histogram on [0, 1], and
Expr = KL(P_circuit || P_Haar) with the Haar distribution integrated over each bin. Lower = more expressive.
QCACNN reports Expr = 0.9185 for its 10-qubit circuit but not its binning; with 75 bins on [0, 1] almost all
10-qubit fidelities fall in the first bin, so we also report a dimension-scaled KL and the effective Haar qubits
log2(1/E[F]), which tests QCACNN's statement that its distribution resembles 9-qubit Haar.

Entangling capability: mean Meyer-Wallach measure Q = 2 (1 - mean_k Tr rho_k^2) over the same output states.

--inputs data additionally repeats both measures for D fixed data inputs (encoded test patches or images) instead
of |0...0>, and reports the mean over inputs; this is the regime the trained circuits actually operate in.

usage: python expr.py --M 500 --out ../logs/expr.jsonl [--inputs data --spec ../datasets/proto/gtsrb_q.csv --D 20]
"""
import argparse, json, time
import numpy as np, torch
import pennylane as qml
import qcacnn as Q
import pqcacnn as P

CIRCUITS = {  # name: (ops, wires, n_params)
    "P6": (P.p6_ops, P.W6, 144),
    "P4": (P.p4_ops, P.W4, 120),
    "QCACNN": (lambda s, w: Q.circuit_ops(s, w), Q.WIRES, Q.N_QPARAMS),
}

def states(ops, wires, w, inputs):
    """Output state vectors (D, 2^n) of the circuit with parameters w for a batch of input states."""
    tape = qml.tape.make_qscript(ops)(inputs, w)
    tape = qml.tape.QuantumScript(tape.operations, [qml.state()])
    (out,) = qml.execute([tape], qml.device("default.qubit", wires=wires))
    return np.asarray(out).reshape(len(inputs), -1)

def kl_haar(F, d, edges):
    p = np.histogram(F, bins=edges)[0] / len(F)
    q = np.diff(1 - (1 - edges) ** (d - 1))                 # Haar: P(F) = (d-1)(1-F)^(d-2), integrated per bin
    m = p > 0
    return float(np.sum(p[m] * np.log(p[m] / np.maximum(q[m], 1e-300))))

def expressibility(psi, bins=75):
    """psi: (M, d) output states for one fixed input. Returns
    - KL to Haar with Sim et al.'s 75 bins on [0, 1] (uninformative for large d: Haar fidelities ~ 1/d all fall in bin 0),
    - KL with 75 bins on [0, 10/d] plus one overflow bin (resolves the Haar peak at any d),
    - effective Haar qubits log2(1 / E[F]) (Haar: E[F] = 1/d), i.e. the Haar dimension with the same mean fidelity."""
    d = psi.shape[1]
    G = np.abs(psi.conj() @ psi.T) ** 2
    F = G[np.triu_indices(len(psi), k=1)]
    std = kl_haar(F, d, np.linspace(0, 1, bins + 1))
    cap = min(1.0, 10.0 / d)
    scaled = kl_haar(F, d, np.append(np.linspace(0, cap, bins + 1), 1.0) if cap < 1 else np.linspace(0, 1, bins + 1))
    return std, scaled, float(np.log2(1.0 / F.mean()))

def meyer_wallach(psi, n):
    """Mean Meyer-Wallach Q over the states in psi (M, 2^n)."""
    t = psi.reshape(len(psi), *([2] * n))
    purity = 0.0
    for k in range(n):
        a = np.moveaxis(t, k + 1, 1).reshape(len(psi), 2, -1)
        rho = a @ a.conj().transpose(0, 2, 1)
        purity = purity + np.real(np.einsum("bij,bji->b", rho, rho))
    return float(np.mean(2 * (1 - purity / n)))

def data_inputs(name, spec, D, seed):
    """D encoded inputs from the test set: random unit-norm colour patches (P6, P4) or softmax-encoded images (QCACNN)."""
    import exp2
    _, _, te_crops, _, _ = exp2.load(spec, 8, False, "rgb")
    X = exp2.to_tensor(te_crops, "rgb")
    rng = np.random.default_rng(seed)
    if name == "QCACNN":
        idx = rng.choice(len(X), D, replace=False)
        return Q.encode(X[idx], 0.1, "sqrt").numpy()
    k = 4 if name == "P6" else 2
    A = P.encode_patches(X[rng.choice(len(X), 64, replace=False)], k, "l2").numpy()
    A = A[np.linalg.norm(A[:, : 3 * k * k], axis=1) > 0]   # skip all-black patches
    return A[rng.choice(len(A), D, replace=False)]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=500); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inputs", default="zero", choices=["zero", "data"])
    ap.add_argument("--spec"); ap.add_argument("--D", type=int, default=20)
    ap.add_argument("--circuits", default="P6,P4,QCACNN"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for name in a.circuits.split(","):
        ops, wires, npar = CIRCUITS[name]; n = len(wires); t0 = time.time()
        rng = np.random.default_rng(a.seed)
        if a.inputs == "zero":
            inp = np.zeros((1, 2 ** n)); inp[0, 0] = 1.0
        else:
            inp = data_inputs(name, a.spec, a.D, a.seed)
        psi = np.stack([states(ops, wires, torch.tensor(rng.random(npar) * 2 * np.pi), inp) for _ in range(a.M)], axis=1)
        ex = np.array([expressibility(psi[i]) for i in range(len(inp))])
        mw = [meyer_wallach(psi[i], n) for i in range(len(inp))]
        res = dict(circuit=name, qubits=n, params=npar, M=a.M, inputs=a.inputs, D=len(inp),
                   expr=round(float(ex[:, 0].mean()), 4), expr_std=round(float(ex[:, 0].std()), 4),
                   expr_scaled=round(float(ex[:, 1].mean()), 4), expr_scaled_std=round(float(ex[:, 1].std()), 4),
                   eff_qubits=round(float(ex[:, 2].mean()), 3), eff_qubits_std=round(float(ex[:, 2].std()), 3),
                   mw=round(float(np.mean(mw)), 4), mw_std=round(float(np.std(mw)), 4),
                   spec=a.spec, seconds=round(time.time() - t0))
        print(json.dumps(res), flush=True)
        with open(a.out, "a") as fh:
            fh.write(json.dumps(res) + "\n")

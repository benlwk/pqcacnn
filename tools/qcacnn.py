"""QCACNN (Qu et al., IEEE IoT J. 12(17):34708-34719, 2025) re-implemented from the paper, plus its CNN and SENet baselines.

No code was published; the circuit is reconstructed from Figs. 3-10 and Sec. III. The reconstruction reproduces the
paper's own resource counts for 10 qubits: 292 quantum parameters (eq. 12), 32 single-qubit + 636 controlled = 668 gates
(eqs. 8-10, Table III). Run with --check to print them.

Training follows Sec. IV-A: 2048 balanced samples per iteration in 8 batches of 256, Adam lr 0.01, 100 iterations,
test set evaluated after each iteration.

usage: python qcacnn.py --model qcacnn|cnn|senet --spec <split.csv> --seed 0 [--amp p|sqrt] [--T 0.1] [--check]
"""
import argparse, csv, math, os, time
import numpy as np, cv2, torch, torch.nn as nn
import pennylane as qml
from sklearn.metrics import f1_score

# ---------------------------------------------------------------- quantum circuit (Figs. 3-10)
FEAT = [f"q{i}" for i in range(8)]                 # feature qubits; q0 = most significant pixel-index bit (low frequency)
QC1, QC0 = "qc1", "qc0"                            # channel qubits: R=|00>, G=|01>, B=|10>
WIRES = [QC1, QC0] + FEAT                          # amplitude index = c * 256 + pixel_index  (eq. 3)
CTRL_STATES = [(0, 0), (1, 0), (0, 1), (1, 1)]     # Fig. 5: (qc1, qc0) control values of CL(Theta_1..4)

def conv_block(t, top, bot):                       # Fig. 3, 10 parameters (Hur et al.); top wire plays "q1"
    qml.RX(t[0], wires=top); qml.RX(t[1], wires=bot)
    qml.RZ(t[2], wires=top); qml.RZ(t[3], wires=bot)
    qml.CRX(t[4], wires=[bot, top]); qml.CRX(t[5], wires=[top, bot])
    qml.RX(t[6], wires=top); qml.RX(t[7], wires=bot)
    qml.RZ(t[8], wires=top); qml.RZ(t[9], wires=bot)

def conv_layer(t, wires):                          # Fig. 4: wires listed top->bottom; Theta_1 on (0,1),(2,3).., Theta_2 on (1,2),(3,4)..
    for i in range(0, len(wires) - 1, 2):
        conv_block(t[0], wires[i], wires[i + 1])
    for i in range(1, len(wires) - 1, 2):
        conv_block(t[1], wires[i], wires[i + 1])

def controlled_conv_layer(t, wires):               # Fig. 5: four conv layers, one per channel-qubit eigenstate; 80 parameters
    for k, (v1, v0) in enumerate(CTRL_STATES):
        qml.ctrl(conv_layer, control=[QC1, QC0], control_values=[v1, v0])(t[k], wires)

def transfer(t, ctrl, tgt):                        # Fig. 6: 2 parameters
    qml.CRZ(t[0], wires=[ctrl, tgt]); qml.PauliX(wires=ctrl); qml.CRX(t[1], wires=[ctrl, tgt])

def pooling(t, r4, r3, r2, r1, r0):                # Fig. 7: info on r3, r1 pooled onto r4, r2, r0; 8 parameters
    transfer(t[0], r3, r4); transfer(t[2], r1, r2)
    transfer(t[1], r3, r2); transfer(t[3], r1, r0)

def qcal(t_cb, t_tr, g1, g0):                      # Fig. 8: global-info qubits g1 ("q1"), g0 ("q0") -> channel qubits; 18 parameters
    conv_block(t_cb, g1, g0)
    transfer(t_tr[0], g0, QC0); transfer(t_tr[1], g1, QC0)
    transfer(t_tr[2], g0, QC1); transfer(t_tr[3], g1, QC1)

N_QPARAMS = 3 * 80 + 2 * 8 + 2 * 18               # = 292 = 53*10 - 238 (eq. 12)

def unpack(w):                                     # flat 292-vector -> per-layer tensors
    shapes = [("ccl", (4, 2, 10)), ("pool", (4, 2)), ("qca_cb", (10,)), ("qca_tr", (4, 2))]
    seq = ["ccl", "pool", "qca_cb", "qca_tr", "ccl", "pool", "qca_cb", "qca_tr", "ccl"]
    out, i, sh = [], 0, dict(shapes)
    for name in seq:
        n = int(np.prod(sh[name])); out.append(w[i:i + n].reshape(sh[name])); i += n
    assert i == N_QPARAMS
    return out

def circuit_ops(state, w):                         # Fig. 10
    ccl1, pool2, cb3, tr3, ccl4, pool5, cb6, tr6, ccl7 = unpack(w)
    qml.StatePrep(state, wires=WIRES, validate_norm=False)
    q = {i: f"q{i}" for i in range(8)}
    controlled_conv_layer(ccl1, [q[7], q[6], q[5], q[4], q[3], q[2], q[1], q[0]])
    pooling(pool2, q[5], q[4], q[2], q[1], q[0]);  qcal(cb3, tr3, q[4], q[1])        # q4, q1 discarded
    controlled_conv_layer(ccl4, [q[7], q[6], q[5], q[3], q[2], q[0]])
    pooling(pool5, q[7], q[6], q[5], q[3], q[2]);  qcal(cb6, tr6, q[6], q[3])        # q6, q3 discarded
    controlled_conv_layer(ccl7, [q[7], q[5], q[2], q[0]])
    return qml.probs(wires=[q[7], q[5], q[2], q[0]])                                 # channel qubits discarded

DEV = qml.device("default.qubit", wires=WIRES)
qcacnn_circuit = qml.QNode(circuit_ops, DEV, interface="torch", diff_method="backprop")

def encode(x, T, amp):                             # Sec. III-B: softmax with temperature over all 3*256 values, eq. (2)-(3)
    B = x.shape[0]
    p = torch.softmax(x.reshape(B, -1).double() / T, dim=1)          # (B, 768), channel-major: R, G, B
    a = p.sqrt() if amp == "sqrt" else p / p.norm(dim=1, keepdim=True)
    return torch.cat([a, torch.zeros(B, 256, dtype=a.dtype, device=a.device)], dim=1)   # channel |11> empty

class QCACNN(nn.Module):
    def __init__(self, T=0.1, amp="p"):
        super().__init__()
        self.T, self.amp = T, amp
        self.qw = nn.Parameter(torch.rand(N_QPARAMS) * 2 * math.pi)   # PennyLane TorchLayer default init U(0, 2pi)
        self.fc = nn.Linear(16, 8)                                     # Sec. III-D: 16 outcome probabilities -> 8 classes
    def forward(self, x):
        probs = qcacnn_circuit(encode(x, self.T, self.amp), self.qw.double())
        return self.fc(probs.float())

# ---------------------------------------------------------------- classical baselines (Figs. 12-13)
class SE(nn.Module):                               # Fig. 13: 2*C^2 parameters
    def __init__(self, c):
        super().__init__(); self.f = nn.Sequential(nn.Linear(c, c, bias=False), nn.ReLU(), nn.Linear(c, c, bias=False), nn.Sigmoid())
    def forward(self, x):
        return x * self.f(x.mean((2, 3)))[:, :, None, None]

class CNN(nn.Module):                              # Fig. 12: 344 conv parameters + the same 16->8 FC (136) = 480
    def __init__(self, se=False, softmax_T=None, act="relu", pool="max"):
        super().__init__()
        self.softmax_T = softmax_T                 # optional: feed the CNN QCACNN's softmax-with-temperature input (test only)
        A = {"relu": nn.ReLU, "tanh": nn.Tanh}[act]          # activation / pooling are not stated in the paper
        P = {"max": nn.MaxPool2d, "avg": nn.AvgPool2d}[pool]
        L = [nn.Conv2d(3, 3, 3, padding=1), A(), P(2)] + ([SE(3)] if se else []) + \
            [nn.Conv2d(3, 4, 3, padding=1), A(), P(2)] + ([SE(4)] if se else []) + \
            [nn.Conv2d(4, 4, 3, padding=1), A(), P(2), nn.Flatten(), nn.Linear(16, 8)]
        self.net = nn.Sequential(*L)
    def forward(self, x):
        if self.softmax_T is not None:
            x = torch.softmax(x.reshape(x.shape[0], -1) / self.softmax_T, dim=1).reshape(x.shape)
        return self.net(x)

# ---------------------------------------------------------------- data (Sec. IV-A-1)
def load(spec, size=16, interp="linear"):
    rows = list(csv.DictReader(open(spec)))
    train, test = {k: [] for k in range(8)}, []
    for r in rows:
        (train[int(r["label"])] if r["split"] == "train" else test).append(r)
    rng = np.random.RandomState(42); bal = []
    for k in range(8):                             # "normalized to 256 entries": discard extras / replicate
        e = train[k]; rng.shuffle(e)
        bal += e[:256] if len(e) >= 256 else [e[i] for i in rng.choice(len(e), 256, replace=True)]
    def img(r):
        im = cv2.imread(r["path"]); x1, y1, x2, y2 = (int(float(r[k])) for k in ("x1", "y1", "x2", "y2"))
        crop = im[y1:y2, x1:x2]; crop = im if crop.size == 0 else crop
        mode = {"linear": cv2.INTER_LINEAR, "area": cv2.INTER_AREA}[interp]
        return cv2.cvtColor(cv2.resize(crop, (size, size), interpolation=mode), cv2.COLOR_BGR2RGB)
    X = lambda rs: torch.tensor(np.stack([img(r) for r in rs]), dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    y = lambda rs: torch.tensor([int(r["label"]) for r in rs])
    return X(bal), y(bal), X(test), y(test)

# ---------------------------------------------------------------- main
def count_resources():
    w = torch.rand(N_QPARAMS, dtype=torch.float64); st = encode(torch.rand(1, 3, 16, 16), 0.1, "p")[0]
    tape = qml.workflow.construct_tape(qcacnn_circuit)(st, w)
    ops = [op for op in tape.operations if not isinstance(op, qml.StatePrep)]
    single = sum(len(op.wires) == 1 for op in ops)
    print(f"[check] quantum params {N_QPARAMS} (paper eq.12: 292) | gates {len(ops)} = {single} single + "
          f"{len(ops) - single} controlled (paper Table III: 668 = 32 + 636)")
    m = QCACNN(); print(f"[check] QCACNN total params {sum(p.numel() for p in m.parameters())} (paper/report: 428)")
    for se in (False, True):
        c = CNN(se); n = sum(p.numel() for p in c.parameters())
        print(f"[check] {'SENet' if se else 'CNN'} params {n} (conv part {n - 136}; paper: CNN 344 + FC, SE adds 50)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qcacnn", choices=["qcacnn", "cnn", "senet"])
    ap.add_argument("--spec"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", default="p", choices=["p", "sqrt"]); ap.add_argument("--T", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.01); ap.add_argument("--check", action="store_true")
    ap.add_argument("--cnn-softmax-input", action="store_true", help="CNN/SENet get QCACNN's softmax(x/T) input")
    ap.add_argument("--act", default="relu", choices=["relu", "tanh"]); ap.add_argument("--pool", default="max", choices=["max", "avg"])
    ap.add_argument("--interp", default="linear", choices=["linear", "area"])
    ap.add_argument("--standardize", action="store_true", help="per-channel mean/std from the training set")
    a = ap.parse_args()
    if a.check:
        count_resources(); raise SystemExit
    torch.manual_seed(a.seed); dev = torch.device("cuda")
    t0 = time.time(); Xtr, ytr, Xte, yte = (t.to(dev) for t in load(a.spec, interp=a.interp))
    if a.standardize:
        mu, sd = Xtr.mean((0, 2, 3), keepdim=True), Xtr.std((0, 2, 3), keepdim=True)
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    model = (QCACNN(a.T, a.amp) if a.model == "qcacnn" else
             CNN(se=a.model == "senet", softmax_T=a.T if a.cnn_softmax_input else None, act=a.act, pool=a.pool)).to(dev)
    print(f"[qcacnn] {a.model} amp={a.amp} T={a.T} act={a.act} pool={a.pool} interp={a.interp} std={a.standardize} spec={os.path.basename(a.spec)} seed={a.seed} | train {tuple(Xtr.shape)} "
          f"test {tuple(Xte.shape)} | params {sum(p.numel() for p in model.parameters())} | data {time.time() - t0:.0f}s", flush=True)
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
    print(f"[qcacnn] done in {time.time() - t0:.0f}s", flush=True)

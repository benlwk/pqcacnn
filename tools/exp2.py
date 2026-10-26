"""Experiment 2: full-class training and image-degradation robustness for P6 / CNN / QCACNN.

  --spec   split CSV (path,x1,y1,x2,y2,label,split); --classes N
  --balance: training balanced to 256/class (QCACNN protocol, 8-class runs); otherwise the full training set is used
  Training: Adam lr 0.01, batch 256 (QCACNN protocol), --epochs E. Final-epoch clean accuracy / macro-F1 reported,
  then the trained model is evaluated on degraded copies of the test set (degradation applied to the sign crop
  before the 16x16 resize, i.e. as a camera/scene effect).

usage: python exp2.py --model p6|cnn|qcacnn --spec ../datasets/proto/gtsrb_q.csv --classes 8 --balance --epochs 100 --seed 0 --out ../logs/results.jsonl
"""
import argparse, csv, json, math, os, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, cv2, torch, torch.nn as nn
from sklearn.metrics import f1_score
import qcacnn as Q
from pqcacnn import PQCACNN

SIZE = 16

def read_crop(r):
    im = cv2.imread(r["path"]); x1, y1, x2, y2 = (int(float(r[k])) for k in ("x1", "y1", "x2", "y2"))
    c = im[y1:y2, x1:x2]
    return im if c.size == 0 else c

COLOUR_SPACES = ["rgb", "lab", "ycrcb", "hsv", "rgby", "hsv4", "lab4"]

def colour(img, cs):
    """uint8 BGR (SIZE x SIZE) -> float32 (C, SIZE, SIZE). 3-channel spaces follow the HQCNN normalisation
    (luminance-like in [0,1], chroma in [-1,1]); 4-channel variants fill the unused channel-qubit state |11>."""
    f = lambda a: a.astype(np.float32)
    if cs in ("rgb", "rgby"):
        rgb = f(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) / 255.0
        ch = [rgb[..., 0], rgb[..., 1], rgb[..., 2]]
        if cs == "rgby":
            ch.append(f(cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[..., 0]) / 255.0)       # luminance Y
    elif cs in ("lab", "lab4"):
        lab = f(cv2.cvtColor(img, cv2.COLOR_BGR2Lab))
        L, a, b = lab[..., 0] / 255.0, (lab[..., 1] - 128) / 127.0, (lab[..., 2] - 128) / 127.0
        ch = [L, a, b] + ([np.sqrt(a ** 2 + b ** 2) / np.sqrt(2)] if cs == "lab4" else [])   # chroma
    elif cs == "ycrcb":
        y = f(cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb))
        ch = [y[..., 0] / 255.0, (y[..., 1] - 128) / 127.0, (y[..., 2] - 128) / 127.0]
    else:                                                                              # hsv / hsv4
        h = f(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)); H, S, V = h[..., 0] / 179.0, h[..., 1] / 255.0, h[..., 2] / 255.0
        ch = [H, S, V] if cs == "hsv" else [V, S, S * np.cos(2 * np.pi * H), S * np.sin(2 * np.pi * H)]   # circular hue
    return np.stack(ch)

def to_tensor(crops, cs="rgb"):
    arr = np.stack([colour(cv2.resize(c, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR), cs) for c in crops])
    return torch.tensor(arr, dtype=torch.float32)

def load(spec, n_classes, balance, cs="rgb", per_class=0, draw=0):
    train, test = {k: [] for k in range(n_classes)}, []
    for r in csv.DictReader(open(spec)):
        (train[int(r["label"])] if r["split"] == "train" else test).append(r)
    if balance:                                                    # QCACNN: discard extras / replicate to 256 per class
        rng = np.random.RandomState(42 + 1000 * draw if per_class else 42); tr = []
        for k in range(n_classes):
            e = train[k]; rng.shuffle(e)
            if per_class: e = e[:per_class]                        # low-data: n distinct images, replicated to 256
            tr += e[:256] if len(e) >= 256 else [e[i] for i in rng.choice(len(e), 256, replace=True)]
    else:
        tr = [r for k in range(n_classes) for r in train[k]]
    with ThreadPoolExecutor(8) as ex:
        tr_crops = list(ex.map(read_crop, tr)); te_crops = list(ex.map(read_crop, test))
    ytr = torch.tensor([int(r["label"]) for r in tr]); yte = torch.tensor([int(r["label"]) for r in test])
    return to_tensor(tr_crops, cs), ytr, te_crops, yte, tr_crops

# ---------------------------------------------------------------- training-time augmentation (on uint8 BGR crops)
# No horizontal flips (they change sign meaning). Photometric ranges are deliberately milder than the test degradations.
def augment(c, rng):
    h, w = c.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-10, 10), rng.uniform(0.9, 1.1))
    M[:, 2] += (rng.uniform(-0.1, 0.1) * w, rng.uniform(-0.1, 0.1) * h)
    f = cv2.warpAffine(c, M, (w, h), borderMode=cv2.BORDER_REPLICATE).astype(np.float32) / 255.0
    f = f * rng.uniform(0.6, 1.4)                                            # brightness
    m = f.mean(); f = (f - m) * rng.uniform(0.7, 1.3) + m                    # contrast
    if rng.random() < 0.3: f = cv2.GaussianBlur(f, (0, 0), sigmaX=rng.uniform(0.3, 1.0))
    if rng.random() < 0.3: f = f + rng.normal(0, rng.uniform(0.01, 0.1), f.shape)
    return (np.clip(f, 0, 1) * 255).astype(np.uint8)

# ---------------------------------------------------------------- degradations (on uint8 BGR crops)
def degrade(c, kind, s, rng):
    f = c.astype(np.float32) / 255.0
    if kind == "noise":      f = f + rng.normal(0, s, f.shape)
    elif kind == "blur":     f = cv2.GaussianBlur(f, (0, 0), sigmaX=s)
    elif kind == "motion":
        k = np.zeros((s, s), np.float32); k[s // 2, :] = 1.0 / s; f = cv2.filter2D(f, -1, k)
    elif kind == "bright":   f = f * s
    elif kind == "contrast": f = (f - f.mean()) * s + f.mean()
    elif kind == "jpeg":
        ok, b = cv2.imencode(".jpg", c, [cv2.IMWRITE_JPEG_QUALITY, int(s)]); return cv2.imdecode(b, cv2.IMREAD_COLOR)
    elif kind == "occlude":
        h, w = c.shape[:2]; side_h, side_w = max(1, int(h * np.sqrt(s))), max(1, int(w * np.sqrt(s)))
        y0, x0 = rng.integers(0, h - side_h + 1), rng.integers(0, w - side_w + 1)
        f = f.copy(); f[y0:y0 + side_h, x0:x0 + side_w] = 0.0
    return (np.clip(f, 0, 1) * 255).astype(np.uint8)

CONDITIONS = [("clean", None)] + [("noise", s) for s in (0.05, 0.1, 0.2)] + [("blur", s) for s in (1.0, 2.0, 3.0)] + \
             [("motion", s) for s in (5, 9, 15)] + [("bright", s) for s in (0.3, 0.6, 1.5)] + [("contrast", s) for s in (0.3, 0.6)] + \
             [("jpeg", s) for s in (30, 10)] + [("occlude", s) for s in (0.1, 0.25)]

# ---------------------------------------------------------------- models
class FixedProj(nn.Module):
    """Untrained random linear map (same matrix for every run), so it adds no trainable parameters."""
    def __init__(self, n_in, n_out, seed=12345):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("W", torch.randn(n_out, n_in, generator=g) / math.sqrt(n_in))
    def forward(self, x):
        return x @ self.W.t()

class StandIn(nn.Module):
    """P6 with its quantum circuit replaced by a classical patch map (same unit-norm patch input, same 6 features/patch,
    same pooling + BatchNorm + Linear head). mlp2: 48->2->6 tanh (116 params < 144 quantum); mlp3: 48->3->6 tanh (171);
    lin6: 48->6 tanh (294); m144: 48->3 without bias (exactly 144 trainable, like the circuit) -> tanh -> fixed random
    3->6 projection -> tanh."""
    def __init__(self, kind, n_classes):
        super().__init__()
        from pqcacnn import encode_patches
        self.encode = encode_patches
        h = {"mlp2": 2, "mlp3": 3}.get(kind)
        if kind == "m144":
            self.map = nn.Sequential(nn.Linear(48, 3, bias=False), nn.Tanh(), FixedProj(3, 6), nn.Tanh())
        else:
            self.map = nn.Sequential(nn.Linear(48, h), nn.Tanh(), nn.Linear(h, 6), nn.Tanh()) if h else \
                       nn.Sequential(nn.Linear(48, 6), nn.Tanh())
        self.pool = POOL                                               # same head as P6: pool x pool grid, BatchNorm, FC
        self.bn, self.fc = nn.BatchNorm1d(6 * POOL * POOL), nn.Linear(6 * POOL * POOL, n_classes)
    def forward(self, x):
        B, g = x.shape[0], x.shape[-1] // 4
        f = self.map(self.encode(x, 4, "l2")[:, :48].float())          # same amplitudes P6 receives (channel |11> padding dropped)
        f = f.reshape(B, g, g, 6).permute(0, 3, 1, 2)
        return self.fc(self.bn(nn.functional.adaptive_avg_pool2d(f, self.pool).flatten(1)))

WIDTH = 3                                                              # set from --width

class CNNW(nn.Module):
    """Width-c classical CNN for parameter-matched comparisons: conv3x3(in->c)+ReLU+maxpool, conv(c->c)+ReLU+maxpool,
    conv(c->c)+ReLU, average-pool to POOL x POOL, FC. Parameters: 18c^2 + 30c + c POOL^2 K + K (3 input channels)."""
    def __init__(self, n_classes, c, in_ch=3, pool=2):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_ch, c, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                 nn.Conv2d(c, c, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                 nn.Conv2d(c, c, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(pool),
                                 nn.Flatten(), nn.Linear(c * pool * pool, n_classes))
    def forward(self, x):
        return self.net(x)

FAST, POOL = False, 2                 # set from --fast / --pool: exact Heisenberg readout for P6; head pooling grid

class Hybrid(nn.Module):
    """Patch front end -> 6-channel map at 1/4 resolution -> small classical CNN backbone (~100K parameters).
    front: p6 (quantum circuit, 144 params), lin6 (classical 48->6 tanh on the same unit-norm patches, 294),
    conv (4x4 stride-4 convolution 3->6 + tanh on raw pixels, 294)."""
    def __init__(self, front, n_classes):
        super().__init__()
        from pqcacnn import encode_patches
        self.front, self.encode = front, encode_patches
        if front == "p6": self.p6 = PQCACNN("P6", "l2", fast=FAST)
        elif front == "lin6": self.map = nn.Sequential(nn.Linear(48, 6), nn.Tanh())
        else: self.map = nn.Sequential(nn.Conv2d(3, 6, 4, stride=4), nn.Tanh())
        cbr = lambda i, o: [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU()]
        self.backbone = nn.Sequential(*cbr(6, 32), *cbr(32, 64), nn.MaxPool2d(2), *cbr(64, 128),
                                      nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, n_classes))
    def forward(self, x):
        B, g = x.shape[0], x.shape[-1] // 4
        if self.front == "p6": f = self.p6.patch_features(x)
        elif self.front == "lin6": f = self.map(self.encode(x, 4, "l2")[:, :48].float()).reshape(B, g, g, 6).permute(0, 3, 1, 2)
        else: f = self.map(x)
        return self.backbone(f)

class GlobalQCACNN(nn.Module):
    """QCACNN's blocks with global amplitude encoding of an s x s image on f = 2 log2(s) feature + 2 channel qubits
    (gradvar.global_ops; f = 8 is the 10-qubit size), sqrt-softmax encoding (T = 0.1), BatchNorm head on the 16 outcome
    probabilities. Used to test QCACNN-style global encoding at 32 x 32 (12 qubits)."""
    def __init__(self, f, n_classes, T=0.1):
        super().__init__()
        import pennylane as qml, gradvar
        self.T = T
        meas = gradvar.global_plan(f)[1]
        def circ(state, w):
            gradvar.global_ops(state, w, f); return qml.probs(wires=meas)
        dev = qml.device("default.qubit", wires=[Q.QC1, Q.QC0] + [f"q{i}" for i in range(f)])
        self.qnode = qml.QNode(circ, dev, interface="torch", diff_method="backprop")
        self.qw = nn.Parameter(torch.rand(gradvar.n_global(f)) * 2 * math.pi)
        self.fc = nn.Sequential(nn.BatchNorm1d(16), nn.Linear(16, n_classes))
    def forward(self, x):
        B = x.shape[0]
        p = torch.softmax(x.reshape(B, -1).double() / self.T, dim=1).sqrt()
        a = torch.cat([p, torch.zeros(B, p.shape[1] // 3, dtype=p.dtype, device=p.device)], dim=1)
        return self.fc(self.qnode(a, self.qw.double()).float())

def build(name, n_classes, in_ch=3):
    if name.startswith("hyb_"):
        return Hybrid(name[4:], n_classes)
    if name == "cnnw":
        return CNNW(n_classes, WIDTH, in_ch, POOL)
    if name == "p8":
        from patchq import P8
        return P8(n_classes, POOL)
    if name.startswith("gqc"):                                         # gqc<f>_bn
        return GlobalQCACNN(int(name[3:].split("_")[0]), n_classes)
    if name in ("stand_mlp2", "stand_mlp3", "stand_lin6", "stand_m144"):
        return StandIn(name.split("_")[1], n_classes)
    if name == "qcacnn_bn":                                            # head control: BatchNorm on the 16 outcome probabilities
        m = Q.QCACNN(T=0.1, amp="sqrt"); m.fc = nn.Sequential(nn.BatchNorm1d(16), nn.Linear(16, n_classes))
        return m
    if name == "p6_nobn":                                              # head control: P6 without BatchNorm
        m = PQCACNN("P6", "l2"); m.bn = nn.Identity()
        if n_classes != 8: m.fc = nn.Linear(24, n_classes)
        return m
    if name == "p6_frozen":                                            # quantum circuit left at its random initialisation
        m = PQCACNN("P6", "l2"); m.qw.requires_grad_(False)
        if n_classes != 8: m.fc = nn.Linear(24, n_classes)
        return m
    if name == "p6":
        m = PQCACNN("P6", "l2", fast=FAST, pool=POOL)
        if n_classes != 8: m.fc = nn.Linear(6 * POOL * POOL, n_classes)
    elif name == "cnn":
        m = Q.CNN()
        if n_classes != 8: m.net[-1] = nn.Linear(16, n_classes)
        if in_ch != 3: m.net[0] = nn.Conv2d(in_ch, 3, 3, padding=1)        # 4-channel colour encodings
        if SIZE != 16: m.net.insert(len(m.net) - 2, nn.AdaptiveMaxPool2d(2))  # keep the 16-feature FC at other sizes
    elif name == "qcacnn":
        assert n_classes == 8, "QCACNN is defined for 8 classes here"
        m = Q.QCACNN(T=0.1, amp="sqrt")
    return m

@torch.no_grad()
def evaluate(model, X, y, n_classes, dev):
    model.eval()
    pred = torch.cat([model(X[i:i + 512].to(dev)).argmax(1).cpu() for i in range(0, len(y), 512)]).numpy()
    yy = y.numpy()
    return 100 * float((pred == yy).mean()), 100 * f1_score(yy, pred, labels=list(range(n_classes)), average="macro", zero_division=0)

# ---------------------------------------------------------------- hardware-noise model (inference only)
# Global depolarising approximation: with two-qubit error rate p and n CNOTs per circuit (full decomposition,
# incl. state preparation), the state becomes F*rho + (1-F)*I/d with F=(1-p)^n. Pauli expectations then scale by F
# and outcome distributions mix with the uniform one. Shot noise: N samples per estimated quantity.
CNOTS_PER_CIRCUIT = {"p6": 1054, "qcacnn": 4694}      # consistent decomposition (see resources); was 7814 for QCACNN
NOISE_P = [0.0, 1e-5, 1e-4, 1e-3]
NOISE_SHOTS = [0, 10000, 1000, 100]              # 0 = exact expectation values

@torch.no_grad()
def evaluate_noisy(model, kind, X, y, n_classes, dev, p, shots, seed=1234):
    from pqcacnn import encode_patches
    import torch.nn.functional as TF
    model.eval(); Fd = (1.0 - p) ** CNOTS_PER_CIRCUIT[kind]
    torch.manual_seed(seed); gen = torch.Generator(device=dev).manual_seed(seed); preds = []
    for i in range(0, len(y), 256):
        xb = X[i:i + 256].to(dev); B = xb.shape[0]
        if kind == "p6":
            out = model.qnode(encode_patches(xb, model.k, model.enc), model.qw.double())
            f = torch.stack(list(out), dim=-1).float() * Fd                    # (B*16, 6) Pauli expectations
            if shots:                                                        # +/-1 outcomes, N shots per expectation
                q = ((1 + f) / 2).clamp(0, 1)
                f = 2 * torch.binomial(torch.full_like(q, float(shots)), q, generator=gen) / shots - 1
            g = 16 // model.k
            f = f.reshape(B, g, g, 6).permute(0, 3, 1, 2)
            logits = model.fc(model.bn(TF.adaptive_avg_pool2d(f, 2).flatten(1)))
        else:
            probs = Q.qcacnn_circuit(Q.encode(xb, model.T, model.amp), model.qw.double()).float()
            probs = Fd * probs + (1 - Fd) / probs.shape[1]
            if shots:
                probs = torch.distributions.Multinomial(total_count=shots, probs=probs.clamp_min(0)).sample() / shots
            logits = model.fc(probs)
        preds.append(logits.argmax(1).cpu())
    pred = torch.cat(preds).numpy(); yy = y.numpy()
    return 100 * float((pred == yy).mean()), 100 * f1_score(yy, pred, labels=list(range(n_classes)), average="macro", zero_division=0)

# ---------------------------------------------------------------- low-precision deployment (inference only, P6)
# Rotation angles: b-bit uniform grid over one full period [0, 4pi) -- exact for controlled rotations too.
# Uplink: patch amplitudes (C*k*k real entries; padding stays 0) on a b-bit grid, renormalised at the edge.
# Downlink: the 6 Pauli expectations per patch on a b-bit grid in [-1, 1]. Head: BN folded into FC, int8 weights.
NONNEG = {"rgb", "rgby", "hsv"}                      # encodings whose amplitudes are all >= 0
QUANT = [("ref", {})] + [(f"ang{b}", dict(ang=b)) for b in (2, 3, 4, 5, 6, 8)] + \
        [(f"up{b}", dict(up=b)) for b in (2, 3, 4, 6, 8)] + [(f"down{b}", dict(down=b)) for b in (2, 3, 4, 6, 8)] + \
        [("head8", dict(head8=True))] + [(f"all{b}", dict(ang=b, up=b, down=b, head8=True)) for b in (4, 6, 8)]

def q_signed(v, b):                                  # mid-tread, 2^b - 1 levels on [-1, 1] (0 representable)
    d = 1.0 / (2 ** (b - 1) - 1); return torch.round(v.clamp(-1, 1) / d) * d

def q_unsigned(v, b):                                # 2^b levels on [0, 1]
    d = 1.0 / (2 ** b - 1); return torch.round(v.clamp(0, 1) / d) * d

def q_angles(w, b):
    per = 4 * math.pi; d = per / 2 ** b
    return torch.remainder(torch.round(torch.remainder(w, per) / d) * d, per)

@torch.no_grad()
def evaluate_quant(model, cs, C, X, y, n_classes, dev, ang=None, up=None, down=None, head8=False):
    from pqcacnn import encode_patches
    import torch.nn.functional as TF
    model.eval(); w = model.qw.double(); bn, fc = model.bn, model.fc
    if ang: w = q_angles(w, ang)
    if head8:                                        # fold BatchNorm (running statistics) into FC; symmetric int8 weights, fp32 bias
        s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        W = fc.weight * s; bias = fc.bias + fc.weight @ (bn.bias - bn.running_mean * s)
        step = W.abs().max() / 127; W = torch.round(W / step) * step
    m = C * model.k * model.k; preds = []
    for i in range(0, len(y), 256):
        xb = X[i:i + 256].to(dev); B = xb.shape[0]
        a = encode_patches(xb, model.k, model.enc)
        if up:
            v = q_unsigned(a[:, :m], up) if cs in NONNEG else q_signed(a[:, :m], up)
            n = v.norm(dim=1, keepdim=True)
            v = torch.where(n > 1e-12, v / n.clamp_min(1e-12), torch.full_like(v, 1 / math.sqrt(m)))
            a = torch.cat([v, a[:, m:]], dim=1)
        f = torch.stack(list(model.qnode(a, w)), dim=-1).float()
        if down: f = q_signed(f, down)
        g = 16 // model.k
        f = TF.adaptive_avg_pool2d(f.reshape(B, g, g, 6).permute(0, 3, 1, 2), 2).flatten(1)
        logits = f @ W.t() + bias if head8 else fc(bn(f))
        preds.append(logits.argmax(1).cpu())
    pred = torch.cat(preds).numpy(); yy = y.numpy()
    return 100 * float((pred == yy).mean()), 100 * f1_score(yy, pred, labels=list(range(n_classes)), average="macro", zero_division=0)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["p6", "cnn", "qcacnn", "stand_mlp2", "stand_mlp3", "stand_lin6", "stand_m144", "p6_frozen", "qcacnn_bn", "p6_nobn",
                                                       "hyb_p6", "hyb_lin6", "hyb_conv", "gqc10_bn", "cnnw", "p8"])
    ap.add_argument("--width", type=int, default=3, help="cnnw: channel width")
    ap.add_argument("--size", type=int, default=16, help="input resolution (sign crop resized to size x size)")
    ap.add_argument("--fast", action="store_true", help="P6: exact Heisenberg readout instead of per-patch simulation")
    ap.add_argument("--pool", type=int, default=2, help="P6 head: average-pool the patch map to pool x pool")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--classes", type=int, required=True); ap.add_argument("--balance", action="store_true")
    ap.add_argument("--per-class", type=int, default=0, help="with --balance: n distinct training images per class "
                    "(drawn per seed), replicated to 256 per class so the number of updates is unchanged")
    ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=256); ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--out", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--cs", default="rgb", choices=COLOUR_SPACES)
    ap.add_argument("--noise-eval", action="store_true", help="also evaluate under depolarising + shot noise (p6/qcacnn)")
    ap.add_argument("--quant-eval", action="store_true", help="also evaluate low-precision angles / links / head (p6)")
    ap.add_argument("--skip-degrade", action="store_true", help="evaluate the clean test set only")
    ap.add_argument("--save", help="save the trained state_dict here")
    ap.add_argument("--augment", action="store_true", help="fresh geometric/photometric augmentation of the training crops every epoch")
    ap.add_argument("--class-balanced", action="store_true", help="sample training images with probability 1/(class size) each epoch")
    a = ap.parse_args()
    SIZE, FAST, POOL, WIDTH = a.size, a.fast, a.pool, a.width
    torch.manual_seed(a.seed); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); t0 = time.time()
    Xtr, ytr, te_crops, yte, tr_crops = load(a.spec, a.classes, a.balance, a.cs, a.per_class, a.seed)
    if not a.augment: del tr_crops
    Xte = to_tensor(te_crops, a.cs)
    Xtr, ytr = Xtr.to(dev), ytr.to(dev)
    model = build(a.model, a.classes, Xtr.shape[1]).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[exp2] {a.tag} model={a.model} classes={a.classes} balance={a.balance} train {tuple(Xtr.shape)} test {tuple(Xte.shape)} "
          f"params {nparam} | data {time.time() - t0:.0f}s", flush=True)
    ntrain = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[exp2] trainable params {ntrain}", flush=True)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=a.lr); crit = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(a.seed)
    if a.class_balanced:
        cw = (1.0 / torch.bincount(ytr.cpu(), minlength=a.classes).clamp_min(1).double())[ytr.cpu()]
    for ep in range(1, a.epochs + 1):
        te = time.time(); model.train(); tl = 0.0
        if a.augment:                                                  # new augmented copy of every training crop
            rng = np.random.default_rng(1000 * a.seed + ep)
            Xtr = to_tensor([augment(c, rng) for c in tr_crops], a.cs).to(dev)
        perm = (torch.multinomial(cw, len(ytr), replacement=True, generator=g) if a.class_balanced
                else torch.randperm(len(ytr), generator=g)).to(dev)
        for i in range(0, len(ytr), a.batch):
            idx = perm[i:i + a.batch]; opt.zero_grad()
            loss = crit(model(Xtr[idx]), ytr[idx]); loss.backward(); opt.step(); tl += loss.item() * len(idx)
        if ep % 10 == 0 or ep == a.epochs:
            acc, f1 = evaluate(model, Xte, yte, a.classes, dev)
            print(f"Epoch {ep:03d}/{a.epochs} | Train Loss: {tl / len(ytr):.4f} | Test Acc: {acc:.2f}% | Test F1: {f1:.2f}% | {time.time() - te:.1f}s", flush=True)
    if a.save:
        os.makedirs(os.path.dirname(a.save) or ".", exist_ok=True); torch.save(model.state_dict(), a.save)
    res = dict(tag=a.tag, model=a.model, spec=os.path.basename(a.spec), classes=a.classes, seed=a.seed, params=nparam, trainable=ntrain, cs=a.cs, augment=a.augment, class_balanced=a.class_balanced, size=a.size, fast=a.fast, pool=a.pool, per_class=a.per_class, width=a.width, cond={})
    tr_acc, tr_f1 = evaluate(model, Xtr.cpu(), ytr.cpu(), a.classes, dev)  # training accuracy (for the generalization gap)
    res["train"] = [round(tr_acc, 2), round(tr_f1, 2)]
    for kind, s in (CONDITIONS[:1] if a.skip_degrade else CONDITIONS):
        rng = np.random.default_rng(1234)
        X = Xte if kind == "clean" else to_tensor([degrade(c, kind, s, rng) for c in te_crops], a.cs)
        acc, f1 = evaluate(model, X, yte, a.classes, dev)
        name = kind if s is None else f"{kind}_{s}"
        res["cond"][name] = [round(acc, 2), round(f1, 2)]
        print(f"[degrade] {name:14s} acc {acc:6.2f}  F1 {f1:6.2f}", flush=True)
    if a.noise_eval and a.model in CNOTS_PER_CIRCUIT:
        res["noise"] = {}
        for p in NOISE_P:
            for n in NOISE_SHOTS:
                acc, f1 = evaluate_noisy(model, a.model, Xte, yte, a.classes, dev, p, n)
                res["noise"][f"p{p:g}_N{n}"] = [round(acc, 2), round(f1, 2)]
                print(f"[noise] p={p:g} shots={n or 'exact'} acc {acc:6.2f}  F1 {f1:6.2f}", flush=True)
    if a.quant_eval and a.model == "p6":
        res["quant"] = {}
        for name, kw in QUANT:
            acc, f1 = evaluate_quant(model, a.cs, Xtr.shape[1], Xte, yte, a.classes, dev, **kw)
            res["quant"][name] = [round(acc, 2), round(f1, 2)]
            print(f"[quant] {name:6s} acc {acc:6.2f}  F1 {f1:6.2f}", flush=True)
    with open(a.out, "a") as fh:
        fh.write(json.dumps(res) + "\n")
    print(f"[exp2] done in {time.time() - t0:.0f}s", flush=True)

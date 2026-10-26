"""Shot-noise accuracy of trained P6 / QCACNN / QCACNN+BN checkpoints over a common grid of shots, for comparison at
equal total shots per image (P6: 16 patches x 3 measurement settings, 2 qubits each -> 48 N shots per image for
N shots per expectation; QCACNN: N shots per image). Uses exp2.evaluate_noisy with p = 0 (no gate noise).

usage: python shots.py --model p6|qcacnn|qcacnn_bn --ckpt ../ckpt/x.pt --spec ../datasets/proto/gtsrb_q.csv --tag x --out ../logs/shots.jsonl
"""
import argparse, json, time
import torch
import exp2

SHOTS = [0, 100, 300, 1000, 3000, 4800, 10000, 14400, 48000, 144000, 480000]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["p6", "qcacnn", "qcacnn_bn"]); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--spec", required=True); ap.add_argument("--cs", default="rgb")
    ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--gate-only", action="store_true", help="skip the shot grid; gate noise with exact expectations only")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); t0 = time.time()
    _, _, te_crops, yte, _ = exp2.load(a.spec, 8, False, a.cs)
    Xte = exp2.to_tensor(te_crops, a.cs)
    model = exp2.build(a.model, 8, Xte.shape[1]); model.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
    model = model.to(dev).eval()
    kind = "p6" if a.model == "p6" else "qcacnn"                # qcacnn_bn: same circuit, BN sits inside model.fc
    per_image = 48 if kind == "p6" else 1
    res = dict(tag=a.tag, model=a.model, cs=a.cs, spec=a.spec.split("/")[-1], ckpt=a.ckpt, shots_per_image_factor=per_image, acc={})
    for n in ([] if a.gate_only else SHOTS):
        acc, f1 = exp2.evaluate_noisy(model, kind, Xte, yte, 8, dev, 0.0, n)
        res["acc"][str(n)] = [round(acc, 2), round(f1, 2)]
        print(f"[shots] {a.model} N={n or 'exact'} (per image {n * per_image}) acc {acc:6.2f} F1 {f1:6.2f}", flush=True)
    res["gate"] = {}                                             # global depolarizing gate noise, exact expectations
    res["cnots"] = exp2.CNOTS_PER_CIRCUIT[kind]
    for p in (1e-5, 1e-4, 1e-3):
        acc, f1 = exp2.evaluate_noisy(model, kind, Xte, yte, 8, dev, p, 0)
        res["gate"][f"{p:g}"] = [round(acc, 2), round(f1, 2)]
        print(f"[gate] {a.model} p={p:g} acc {acc:6.2f} F1 {f1:6.2f}", flush=True)
    res["seconds"] = round(time.time() - t0)
    with open(a.out, "a") as fh:
        fh.write(json.dumps(res) + "\n")

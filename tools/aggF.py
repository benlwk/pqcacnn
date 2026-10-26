"""Summarize the runs: retrained clean accuracy (vs paper), QCACNN noise protocol, expressibility.
usage: python aggF.py ../logs"""
import json, sys, collections, numpy as np
L = sys.argv[1]
def rows(f):
    try: return [json.loads(l) for l in open(f"{L}/{f}") if l.strip()]
    except FileNotFoundError: return []
ms = lambda v: f"{np.mean(v):5.1f} ± {np.std(v):3.1f} (n={len(v)})"
key = lambda r: (r["spec"].split("_")[0], r["model"], r["cs"])

print("== retrained clean accuracy / F1 (paper: P6 rgb 96.0/82.7, ycrcb 95.9/85.3, QCACNN 67.5/55.4)")
g = collections.defaultdict(list)
for r in rows("train.jsonl"): g[key(r)].append(r["cond"]["clean"])
for k in sorted(g): print(k, "acc", ms([v[0] for v in g[k]]), "F1", ms([v[1] for v in g[k]]))

print("== QCACNN noise protocol (acc; F1 in brackets); inference-time channels after encoding and after first QCAL")
g = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows("noise_qc.jsonl"):
    for c, v in r["noise"].items(): g[key(r)][c].append(v)
for k in sorted(g):
    print(k, " | ".join(f"{c} {np.mean([v[0] for v in g[k][c]]):.1f} [{np.mean([v[1] for v in g[k][c]]):.1f}]" for c in g[k]), f"(n={len(g[k]['clean'])})")

print("== expressibility (KL to Haar, lower = more expressive) / Meyer-Wallach")
for r in rows("expr.jsonl"):
    print(f"{r['circuit']:7s} {r['qubits']:2d}q inputs={r['inputs']:4s} {str(r['spec']).split('/')[-1]:12s} "
          f"Expr {r['expr']:.4f} ± {r['expr_std']:.4f}  scaled {r.get('expr_scaled', float('nan')):.4f} ± {r.get('expr_scaled_std', float('nan')):.4f}  "
          f"eff.qubits {r.get('eff_qubits', float('nan')):.2f}  MW {r['mw']:.4f} ± {r['mw_std']:.4f}  ({r['seconds']}s)")

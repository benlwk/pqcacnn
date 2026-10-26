"""Fig.: gradient variance and feature concentration vs image size, global vs patch-wise encoding (from gradvar.jsonl)."""
import json, sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC, OUT = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(SRC) if l.strip()]
BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#1f1f1e", "#6b6a64", "#e4e3dd"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6})

def series(mode, key, name, enc="l2"):
    pts = sorted((r["size"], r[key], r["qubits"]) for r in rows
                 if r["mode"] == mode and r["circuit"].startswith(name) and r["enc"] == enc
                 and (name != "G" or r["circuit"][1:].isdigit()))
    return [np.array(v) for v in zip(*pts)] if pts else (np.array([]),) * 3

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45), constrained_layout=True)
panels = [("circuit", "var_z", r"Var$_{\theta,x}[\partial\langle Z\rangle/\partial\theta_k]$", "(a) Gradient variance"),
          ("feat", "featvar_z", r"Var$_{x}[\langle Z\rangle]$", "(b) Feature variance across images")]
for ax, (mode, key, ylabel, title) in zip(axes, panels):
    s, v, n = series(mode, key, "G")
    ax.plot(s, v, "-o", color=BLUE, lw=2, ms=5, mec="white", mew=1.0, label="Global encoding (QCACNN blocks)", zorder=3)
    for si, vi, ni in zip(s, v, n):
        ax.annotate(f"{ni}q", (si, vi), textcoords="offset points", xytext=(5, 3), fontsize=7, color=MUTED)
    if len(s) >= 3:                                                    # exponential fit over the last points
        k = n >= 10; b = np.polyfit(n[k], np.log2(v[k]), 1)[0] if k.sum() >= 2 else np.nan
        print(f"{key}: global fit exponent {b:.2f} per qubit (n >= 10); points {list(zip(n.tolist(), v.tolist()))}")
    qs, qv, _ = series(mode, key, "QCACNN", "sqrtsoftmax")
    if len(qs):
        ax.plot(qs, qv, "*", color=INK, ms=9, mec="white", mew=0.8, label="QCACNN [8] (10q)", zorder=4)
    for name, col, mk, ls, lab in (("P6", ORANGE, "s", "--", "Patch-wise P6 (6q)"), ("P4", AQUA, "^", ":", "Patch-wise P4 (4q)")):
        ps, pv, _ = series(mode, key, name)
        if len(ps):
            ax.plot(ps, pv, ls, marker=mk, color=col, lw=2, ms=5, mec="white", mew=1.0, label=lab, zorder=3)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks([4, 8, 16, 32, 64, 128]); ax.set_xticklabels(["4", "8", "16", "32", "64", "128"])
    ax.set_xlabel("Image side (pixels)"); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=8, color=INK, loc="left")
    ax.grid(True, which="major", color=GRID, lw=0.5); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
axes[0].legend(fontsize=6.5, frameon=False, loc="lower left")
fig.savefig(OUT); fig.savefig(OUT.replace(".pdf", ".png"), dpi=200)
print("saved", OUT)

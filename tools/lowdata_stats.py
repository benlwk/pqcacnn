"""Low-data study: test accuracy, generalization gap (train - test) and paired comparisons of P6 against each model.

Seeds are paired: seed s draws the same training subset for every model. Reports mean +- std over seeds, the mean
paired difference P6 - model, and a two-sided exact sign-flip permutation test on the paired differences
(no distributional assumption; exact for 10 seeds = 1024 sign patterns).

usage: python lowdata_stats.py lowdata2.jsonl
"""
import json, sys, itertools, statistics as st, collections

rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
MODELS = ["p6", "cnn", "stand_mlp2", "stand_m144", "stand_mlp3", "stand_lin6"]
NAME = {"p6": "P6 (144q)", "cnn": "CNN (480)", "stand_mlp2": "map 116", "stand_m144": "map 144",
        "stand_mlp3": "map 171", "stand_lin6": "map 294"}
acc, gap = collections.defaultdict(dict), collections.defaultdict(dict)
for r in rows:
    k = (r["spec"].split("_")[0], r["per_class"] or 999, r["model"])
    acc[k][r["seed"]] = r["cond"]["clean"][0]
    gap[k][r["seed"]] = r["train"][0] - r["cond"]["clean"][0]

def perm_p(d):
    """Exact two-sided sign-flip test on paired differences d."""
    obs = abs(sum(d)); n = 0; hits = 0
    for signs in itertools.product((1, -1), repeat=len(d)):
        n += 1; hits += abs(sum(s * x for s, x in zip(signs, d))) >= obs - 1e-12
    return hits / n

ms = lambda v: f"{st.mean(v):5.1f}±{st.pstdev(v):4.1f}"
for ds in ("tsrd", "gtsrb"):
    print(f"\n=== {ds.upper()}: test accuracy (mean±std over seeds) | generalization gap train-test")
    print(f"{'n':>4s} " + " ".join(f"{NAME[m]:>13s}" for m in MODELS))
    for n in (5, 10, 25, 50, 100):
        line = [ms(list(acc[(ds, n, m)].values())) if acc[(ds, n, m)] else "" for m in MODELS]
        gl = [f"{st.mean(gap[(ds, n, m)].values()):5.1f}" if gap[(ds, n, m)] else "" for m in MODELS]
        print(f"{n:4d} " + " ".join(f"{x:>13s}" for x in line))
        print(f"{'gap':>4s} " + " ".join(f"{x:>13s}" for x in gl))
    print(f"--- {ds.upper()}: paired difference P6 - model (pp), exact sign-flip p-value")
    for n in (5, 10, 25, 50, 100):
        out = []
        for m in MODELS[1:]:
            a, b = acc[(ds, n, "p6")], acc[(ds, n, m)]
            seeds = sorted(set(a) & set(b))
            if len(seeds) < 3: out.append(""); continue
            d = [a[s] - b[s] for s in seeds]
            out.append(f"{st.mean(d):+5.1f} p={perm_p(d):.3f}")
        print(f"{n:4d} " + " ".join(f"{NAME[m]+':':>9s} {o:<15s}" for m, o in zip(MODELS[1:], out)))
full = [r for r in rows if not r["per_class"]]
for ds in ("tsrd", "gtsrb"):
    v = [r["cond"]["clean"][0] for r in full if r["spec"].startswith(ds)]
    if v: print(f"map 144 at full data, {ds}: {ms(v)} (n={len(v)})")

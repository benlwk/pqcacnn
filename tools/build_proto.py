# Build QCACNN-protocol split specs (official train/test splits), and check exact-duplicate leakage.
# Output CSV columns: path,x1,y1,x2,y2,label,split
import os, csv, hashlib, collections, pandas as pd
R = os.environ.get("HQCNN_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); out = os.path.join(R, "datasets", "proto"); os.makedirs(out, exist_ok=True)
TSRD_MAP = {5: 0, 10: 1, 28: 2, 37: 3, 52: 4, 54: 5, 55: 6, 56: 7}
GTSRB_MAP = {3: 0, 10: 1, 12: 2, 17: 3, 22: 4, 26: 5, 33: 6, 40: 7}     # QCACNN Table I
def tsrd(ann, img_dir, split):
    rows = []
    for line in open(ann):
        p = line.strip().rstrip(";").split(";")
        if len(p) >= 8 and int(p[7]) in TSRD_MAP:
            rows.append([os.path.join(img_dir, p[0]), p[3], p[4], p[5], p[6], TSRD_MAP[int(p[7])], split])
    return rows
def gtsrb(csv_path, split):
    df = pd.read_csv(csv_path); df = df[df.ClassId.isin(GTSRB_MAP)]
    return [[os.path.join(R, "raw_gtsrb", r.Path), r["Roi.X1"], r["Roi.Y1"], r["Roi.X2"] + 1, r["Roi.Y2"] + 1, GTSRB_MAP[int(r.ClassId)], split] for _, r in df.iterrows()]
specs = {
  "tsrd_q.csv": tsrd(f"{R}/raw/TSRD-Train Annotation/TsignRecgTrain4170Annotation.txt", f"{R}/raw/tsrd-train", "train")
              + tsrd(f"{R}/raw/TSRD-Test Annotation/TsignRecgTest1994Annotation.txt", f"{R}/raw/TSRD-Test", "test"),
  "gtsrb_q.csv": gtsrb(f"{R}/raw_gtsrb/Train.csv", "train") + gtsrb(f"{R}/raw_gtsrb/Test.csv", "test"),
}
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
for name, rows in specs.items():
    with open(os.path.join(out, name), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["path", "x1", "y1", "x2", "y2", "label", "split"]); w.writerows(rows)
    cnt = {s: collections.Counter(r[5] for r in rows if r[6] == s) for s in ("train", "test")}
    missing = sum(not os.path.exists(r[0]) for r in rows)
    tr = {md5(r[0]) for r in rows if r[6] == "train"}; te = [md5(r[0]) for r in rows if r[6] == "test"]
    print(f"{name}: train per-class {[cnt['train'][k] for k in range(8)]}, test per-class {[cnt['test'][k] for k in range(8)]} "
          f"(test total {len(te)}), missing files {missing}, test images with identical copy in train: {sum(h in tr for h in te)}", flush=True)
print("PROTO BUILT")

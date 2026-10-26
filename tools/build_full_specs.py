# Full-class split specs on the official splits: TSRD (58 classes), GTSRB (43 classes).
import os, csv, collections, pandas as pd
R = os.environ.get("HQCNN_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); out = os.path.join(R, "datasets", "proto")
def tsrd(ann, d, split):
    rows = []
    for line in open(ann):
        p = line.strip().rstrip(";").split(";")
        if len(p) >= 8: rows.append([os.path.join(d, p[0]), p[3], p[4], p[5], p[6], int(p[7]), split])
    return rows
def gtsrb(path, split):
    df = pd.read_csv(path)
    return [[os.path.join(R, "raw_gtsrb", r.Path), r["Roi.X1"], r["Roi.Y1"], r["Roi.X2"] + 1, r["Roi.Y2"] + 1, int(r.ClassId), split] for _, r in df.iterrows()]
specs = {"tsrd_full.csv": tsrd(f"{R}/raw/TSRD-Train Annotation/TsignRecgTrain4170Annotation.txt", f"{R}/raw/tsrd-train", "train")
                          + tsrd(f"{R}/raw/TSRD-Test Annotation/TsignRecgTest1994Annotation.txt", f"{R}/raw/TSRD-Test", "test"),
         "gtsrb_full.csv": gtsrb(f"{R}/raw_gtsrb/Train.csv", "train") + gtsrb(f"{R}/raw_gtsrb/Test.csv", "test")}
for name, rows in specs.items():
    with open(os.path.join(out, name), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["path", "x1", "y1", "x2", "y2", "label", "split"]); w.writerows(rows)
    c = collections.Counter((r[6], r[5]) for r in rows)
    ncls = len({r[5] for r in rows}); missing = sum(not os.path.exists(r[0]) for r in rows)
    print(f"{name}: classes {ncls}, train {sum(r[6]=='train' for r in rows)}, test {sum(r[6]=='test' for r in rows)}, missing files {missing}, "
          f"min/max train per class {min(c[('train',k)] for k in range(ncls))}/{max(c[('train',k)] for k in range(ncls))}")

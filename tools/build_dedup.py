# TSRD QCACNN-protocol split with every test image that has a byte-identical copy among the training images removed.
import os, csv, hashlib
R = os.path.join(os.environ.get("HQCNN_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets", "proto")
rows = list(csv.DictReader(open(os.path.join(R, "tsrd_q.csv"))))
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
train = {md5(r["path"]) for r in rows if r["split"] == "train"}
keep = [r for r in rows if r["split"] == "train" or md5(r["path"]) not in train]
with open(os.path.join(R, "tsrd_qd.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(keep)
nte = sum(r["split"] == "test" for r in rows); nkeep = sum(r["split"] == "test" for r in keep)
print(f"tsrd_qd.csv: test {nte} -> {nkeep} (removed {nte - nkeep} duplicates of training images)")

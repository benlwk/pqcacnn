# PQCACNN

Code for *A Scalable Patch-Wise Quantum Channel-Attention CNN for Traffic Sign Recognition in the Internet of Vehicles*.

PQCACNN applies QCACNN's channel-aware quantum blocks to small colour patches. One shared 6-qubit (P6) or 8-qubit (P8) circuit processes every patch, and a small classical head classifies the resulting feature map.

## Setup

```bash
pip install -r requirements.txt
export HQCNN_ROOT=$(pwd)        # repository root (default: parent of tools/)
```

Experiments were run with Python 3.12, PyTorch 2.14, PennyLane 0.45, OpenCV 4.13 and scikit-learn 1.8.

## Data

Download TSRD and GTSRB (Kaggle: `khanhvudo72/traffic-sign-recogntion-database`, `meowmeowmeowmeowmeow/gtsrb-german-traffic-sign`) and unzip them into `raw/` and `raw_gtsrb/`. Then build the split files:

```bash
cd tools
python build_proto.py        # 8-class TSRD / GTSRB (QCACNN protocol)
python build_full_specs.py   # all 58 / 43 classes
python build_dedup.py        # duplicate-free TSRD test set
```

## Training

All commands are run from `tools/`. `exp2.py` trains one model with one seed, reports the final-epoch accuracy and macro-F1, and appends a JSON record to `--out`.

```bash
python exp2.py --model p6 --spec ../datasets/proto/gtsrb_q.csv --classes 8 --balance \
               --epochs 100 --seed 0 --out ../logs/results.jsonl --tag p6_gtsrb_s0 --save ../ckpt/p6_gtsrb_s0.pt
```

Useful options:

| Option | Meaning |
|---|---|
| `--model` | `p6`, `p8`, `qcacnn`, `qcacnn_bn`, `gqc10_bn` (12-qubit global encoding), `cnn`, `cnnw`, `stand_mlp2`, `stand_mlp3`, `stand_lin6`, `stand_m144`, `p6_frozen`, `p6_nobn`, `hyb_p6`, `hyb_lin6`, `hyb_conv` |
| `--cs` | colour encoding: `rgb`, `lab`, `ycrcb`, `hsv`, `rgby`, `lab4`, `hsv4` |
| `--pool` | head pooling grid q (2 or 4) |
| `--size` | input resolution (16, 32, 48) |
| `--fast` | exact Heisenberg-picture readout for P6 (same results, much faster) |
| `--per-class n` | low-data regime: n distinct images per class |
| `--skip-degrade` | clean test set only (otherwise all image degradations are evaluated) |
| `--quant-eval` | low-precision evaluation of P6 |
| `--save` | save the trained model for the noise, shot, precision and interpretability scripts |

The paper uses seeds 0–4 (0–9 for the low-data study). Full-class runs use `--spec ../datasets/proto/{tsrd,gtsrb}_full.csv --classes 58|43` without `--balance`, with 100 epochs on TSRD and 30 on GTSRB.

## Reproducing the paper

| Paper result | Command (from `tools/`) |
|---|---|
| Resources (Table I), simulation cost | `python pqcacnn.py --check`, `python qcacnn.py --check`, `python flops.py` |
| Quantum models (Table II) | `exp2.py --model p6 / p8 / qcacnn / qcacnn_bn`; QCACNN with p/‖p‖₂ amplitudes: `python qcacnn.py --model qcacnn --amp p --spec ...`; P4 and sqrt-softmax encoding: `python pqcacnn.py --arch P4|P6 --enc l2|sqrtsoftmax --spec ...` |
| Duplicate-free TSRD | `exp2.py ... --spec ../datasets/proto/tsrd_qd.csv` |
| Gradient and feature variance (Fig. 5) | `python gradvar.py --mode circuit --circuit G8 --size 16 --M 1000 --out ../logs/gradvar.jsonl` (circuits `G4`–`G14`, `QCACNN`, `P6`, `P4`); `--mode feat` for feature variance; `--mode loss --model p6` for loss gradients; plot: `python make_fig_bp.py ../logs/gradvar.jsonl fig_bp.pdf` |
| Expressibility and entanglement | `python expr.py --M 500 --inputs data --D 20 --spec ../datasets/proto/gtsrb_q.csv --out ../logs/expr.jsonl` |
| Resolution, all classes, backbones (Tables III, VI) | `exp2.py --model p6 --size 32 --fast --pool 2`, `--model p8 --pool 4 --size 32`, `--model gqc10_bn --size 32`, `--model hyb_p6 --size 32 --fast` |
| Gate noise and shots (Tables IV, V) | `python shots.py --model p6 --ckpt ../ckpt/p6_gtsrb_s0.pt --spec ../datasets/proto/gtsrb_q.csv --tag x --out ../logs/shots.jsonl` |
| Single-qubit noise (Table S2) | `python noise_qc.py --model p6 --ckpt ../ckpt/p6_gtsrb_s0.pt --spec ../datasets/proto/gtsrb_q.csv --tag x --out ../logs/noise_qc.jsonl` |
| Low precision (Table VII) | `exp2.py --model p6 --quant-eval --save ...`, then `python qfollow.py --ckpt ../ckpt/p6_gtsrb_s0.pt --cs rgb --spec ../datasets/proto/gtsrb_q.csv --tag x --out ../logs/qfollow.jsonl` |
| Image degradation, colour encodings (Tables VIII, IX) | `exp2.py --model p6` or `--model cnn`, without `--skip-degrade`, with `--cs` for the colour space |
| Interpretability (Fig. 6) | `python xai.py --ckpt ../ckpt/p6_gtsrb_s0.pt --cs rgb --spec ../datasets/proto/gtsrb_q.csv --tag x --out ../logs/xai.jsonl` |
| Classical references (Table X) | `exp2.py --model stand_lin6 / stand_mlp3 --pool 4`, `--model cnnw --width 5 --pool 4`, `--model p6_frozen`; SENet: `python qcacnn.py --model senet --spec ...` |
| Low data (Table XI) | `exp2.py --model p6 --fast --per-class 5 --seed 0 ...` (also `cnn`, `stand_m144`, `stand_lin6`); paired sign-flip test: `python lowdata_stats.py ../logs/lowdata.jsonl` |
| Summary of clean accuracy, noise protocol, expressibility | `python aggF.py ../logs` (reads `train.jsonl`, `noise_qc.jsonl` and `expr.jsonl`) |

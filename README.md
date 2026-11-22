# STAF-Net: Spatial-Temporal Aware Fusion Network

Binary video deepfake detection on **DFDC** using two frozen-or-LoRA-adapted
foundation models: **DINOv3 ViT-S/16** (spatial) and **V-JEPA 2.1 ViT-B/16**
(spatiotemporal), combined by adaptive gated fusion.

Course: Pengolahan Citra Digital dan Visi Komputer (CIF61339) · Hardware: 1 × A100 80 GB · GPU budget: 4 h

---

## Research questions

1. How effective are frozen DINOv3 and V-JEPA features for DFDC deepfake detection?
2. Does supervised **LoRA adaptation** improve each encoder over frozen features?
3. Do combined spatial + spatiotemporal features beat either encoder alone?
4. Does **gated fusion** beat simple concatenation?

A negative result is a valid outcome. V-JEPA is video-pretrained, not a deepfake
detector, and the fusion gate is an adaptive weighting — not a causal explanation.

---

## Architecture

```
clip (16 frames, one ~2 s window)
   │
   ├── frames 0,5,10,15 ──> DINOv3 ViT-S/16 @224 ──> z_D (384) ──┐
   │        frozen | LoRA                                        ├──> concat / gated fusion ──> MLP ──> real/fake logit
   └── all 16 frames ─────> V-JEPA 2.1 ViT-B/16 @384 ──> z_J (768) ┘
            frozen | LoRA
```

| Component | Model | Params | Input |
|---|---|---:|---|
| Spatial encoder | DINOv3 ViT-S/16 | 21M | 4 frames, 224×224, layout `(B,T,C,H,W)` |
| Spatiotemporal encoder | V-JEPA 2.1 ViT-B/16 | 80M | 16 frames, 384×384, layout `(B,C,T,H,W)` |
| Adaptation | LoRA r=8, α=16 | small | attention projections |
| Fusion | projection → 256 → gate → MLP | < 1M | pooled vectors only |

Fusion formulas:

```
h_D = GELU(W_D z_D + b_D)                    384 → 256
h_J = GELU(W_J z_J + b_J)                    768 → 256
g   = sigmoid(W_g [h_D || h_J] + b_g)        512 → 256
h   = g * h_D + (1 - g) * h_J                gated
logit = MLP(h)                               256 → 128 → 1
```

---

## Dataset

**DFDC** (Kaggle, `deepfake-detection-challenge`), binary `REAL=0` / `FAKE=1`.

Target subset: 1,500 videos (train 1050, val 224, test 226), balanced real/fake.
Splits are assigned per **source group** — a fake video, its `original` real video,
and every known derivative stay in the same split. DFDC metadata does not
guarantee identity-disjoint splits; this is reported as a limitation.

DFDC ships **video-level labels only**. There are no spatial or temporal mask
targets, so this project does classification only — no manipulation localization.

---

## Two-stage training

**Stage I — branch adaptation (GPU heavy).**
Each branch is trained independently: frozen baseline first, then supervised LoRA
with a temporary binary head. Both frozen and adapted embeddings are cached.

**Stage II — fusion and ablations (CPU/light GPU).**
Both encoders frozen. Classifiers train on cached per-video embeddings.

Caching is what makes the ablation matrix affordable: 1,500 videos × 4 variants
is roughly 13 MB of vectors, not a GPU workload.

---

## Ablation matrix

| ID | DINOv3 | V-JEPA | Classifier | Question |
|---|---|---|---|---|
| A1 | frozen | — | spatial MLP | spatial baseline |
| A2 | — | frozen | video MLP | video baseline |
| A3 | LoRA | — | spatial MLP | does DINO adaptation help? |
| A4 | — | LoRA | video MLP | does V-JEPA adaptation help? |
| A5 | frozen | frozen | concat | complementarity, unadapted |
| A6 | frozen | frozen | gated | does gating help unadapted? |
| A7 | LoRA | LoRA | concat | complementarity, adapted |
| **A8** | **LoRA** | **LoRA** | **gated** | **proposed STAF-Net** |

Key comparisons: A1↔A3, A2↔A4, A5↔A6, A7↔A8, and {A3, A4} ↔ A8.
All runs share identical video IDs, split manifest, test clips, and head protocol.

---

## Repository layout

```text
STAF-Net/
├── configs/
│   ├── dataset.yaml          # DFDC subset, sampling, face crop, split protocol
│   ├── dino_lora.yaml        # Stage I spatial branch
│   ├── vjepa_lora.yaml       # Stage I video branch
│   └── fusion.yaml           # Stage II fusion + ablation driver
├── data/                     # git-ignored contents
│   ├── raw/                  # DFDC videos
│   ├── manifests/            # manifest.csv, split_summary.json
│   ├── clips/                # cached face crops per clip
│   └── features/             # cached embeddings, 4 variants
├── src/
│   ├── contracts.py          # frozen tensor contract v0.2
│   ├── data/
│   │   ├── build_manifest.py # DFDC metadata -> manifest + source-group split
│   │   ├── face_tracking.py  # decode, 16-frame window, detect, track, crop, cache
│   │   ├── video_dataset.py  # clip mode and feature mode datasets
│   │   └── transforms.py     # per-branch normalization + mild augmentation
│   ├── models/
│   │   ├── dino_branch.py    # DinoBranch (Person 1)
│   │   ├── vjepa_branch.py   # VJepaBranch (Person 2)
│   │   └── stafnet.py        # STAFNet fusion + A1-A8 switches (Person 3)
│   ├── utils/
│   │   ├── lora.py           # shared LoRA helper, target inspection
│   │   └── metrics.py        # AUROC, F1, EER, bootstrap CI
│   ├── train_dino.py         # Stage I spatial entry point
│   ├── train_vjepa.py        # Stage I video entry point
│   ├── extract_features.py   # cache variant embeddings
│   ├── train_fusion.py       # Stage II + ablations
│   └── evaluate.py           # final held-out test evaluation
├── scripts/
│   └── smoke_branches.py     # dataset-free contract + branch smoke test
├── checkpoints/              # git-ignored
├── results/
│   ├── metrics/              # git-ignored json
│   └── plots/                # git-ignored png
├── notebooks/
│   ├── 01_eda.ipynb          # DFDC subset EDA
│   └── 02_error_analysis.ipynb
├── docs/
│   ├── cache_format.md       # cached feature schema + invariants
│   └── open_decisions.md     # unresolved decisions blocking parallel work
└── requirements.txt
```

---

## Quickstart

```bash
pip install -r requirements.txt

# 0. verify the contract without any dataset or GPU
python scripts/smoke_branches.py --contracts-only
python scripts/smoke_branches.py --branch dino --batch-size 2

# 1. dataset (once, before the GPU session)
kaggle competitions download deepfake-detection-challenge -f train_sample_videos.zip -p data/raw
python -m src.data.build_manifest --config configs/dataset.yaml
python -m src.data.face_tracking --config configs/dataset.yaml

# 2. Stage I: frozen baselines, then LoRA
python -m src.train_dino  --config configs/dino_lora.yaml   --frozen
python -m src.train_dino  --config configs/dino_lora.yaml
python -m src.train_vjepa --config configs/vjepa_lora.yaml  --frozen
python -m src.train_vjepa --config configs/vjepa_lora.yaml

# 3. cache the four feature variants
python -m src.extract_features --variant dino_frozen  --checkpoint checkpoints/dino_frozen.pt
python -m src.extract_features --variant dino_lora    --checkpoint checkpoints/dino_lora.pt
python -m src.extract_features --variant vjepa_frozen --checkpoint checkpoints/vjepa_frozen.pt
python -m src.extract_features --variant vjepa_lora   --checkpoint checkpoints/vjepa_lora.pt

# 4. Stage II: ablations A1-A8
for exp in A1 A2 A3 A4 A5 A6 A7 A8; do
  python -m src.train_fusion --config configs/fusion.yaml --experiment $exp
done

# 5. final held-out test evaluation
python -m src.evaluate --config configs/fusion.yaml --checkpoint checkpoints/fusion_A8.pt --split test
```

---

## Results

| Experiment | AUROC ↑ | F1 ↑ | Accuracy ↑ | Train time |
|---|---:|---:|---:|---:|
| A1 Frozen DINO | — | — | — | — |
| A2 Frozen V-JEPA | — | — | — | — |
| A3 DINO LoRA | — | — | — | — |
| A4 V-JEPA LoRA | — | — | — | — |
| A5 Frozen concat | — | — | — | — |
| A6 Frozen gated | — | — | — | — |
| A7 Adapted concat | — | — | — | — |
| **A8 STAF-Net** | — | — | — | — |

Primary metric: video-level AUROC. Threshold and checkpoint selection use
validation data only; the test split is evaluated once.

---

## Team

| Role | Owns | Deliverables |
|---|---|---|
| **Person 1** — spatial | `dino_branch.py`, `train_dino.py`, half of `src/data/` | `dino_frozen`, `dino_lora` + checkpoints |
| **Person 2** — video | `vjepa_branch.py`, `train_vjepa.py`, half of `src/data/` | `vjepa_frozen`, `vjepa_lora` + checkpoints |
| **Person 3** — fusion | `stafnet.py`, `train_fusion.py`, `evaluate.py`, `metrics.py`, `contracts.py` | A1-A8 results, tables, plots |

All three agree on the manifest, preprocessing contract, seeds, and feature
schema first (`src/contracts.py` + `docs/cache_format.md`). GPU-heavy jobs need
exclusive A100 access — schedule them.

---

## Status

Skeleton stage. `src/contracts.py` (v0.2) and `scripts/smoke_branches.py` are
implemented and verified; every model, dataset, and training module is a
documented placeholder with its contract, hyperparameters, and TODO list.

## Limitations to report

- DFDC metadata does not guarantee identity-disjoint splits; grouping is by
  `original` linkage only.
- One clip per video; a clip from a fake-labeled video may not visibly contain
  manipulation.
- 4-hour GPU budget; frozen baselines and A1/A2/A5/A6 have priority.
- Frozen third-party weights keep their own licenses (see `docs/open_decisions.md`).

## License

Code: MIT (see `LICENSE`). Backbone weights: DINOv3 and V-JEPA 2.1 keep their own terms.

# STARD: Spatial-Temporal Deepfake & Physical Spoof Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**STARD** is a unified deep learning framework designed to detect facial video manipulations—including generative **Deepfakes** and **Physical Spoofs** (e.g., photo prints or screen replays)—by adaptively fusing fine-grained **spatial artifacts** with long-term **temporal dynamics**.

---

## 🏗️ Architecture Overview

Rather than analyzing frames in isolation, STARD employs a dual-backbone vision transformer pipeline coupled with an adaptive gated fusion mechanism:

```
                            ┌──────────────────────────┐
                            │       Input Video        │
                            │ 16 frames (Facial Crops) │
                            └────────────┬─────────────┘
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   │                                           │
                   ▼                                           ▼
    ┌──────────────────────────────┐           ┌──────────────────────────────┐
    │        DINOv3 ViT-B          │           │       V-JEPA 2.1 ViT-B       │
    │          (Spatial)           │           │          (Temporal)          │
    │ Per-frame patch embeddings,  │           │ Motion context & temporal    │
    │ local blending artifacts     │           │ correspondence               │
    └──────────────┬───────────────┘           └──────────────┬───────────────┘
                   │                                           │
                   └─────────────────────┬─────────────────────┘
                                         │
                                         ▼
                            ┌──────────────────────────┐
                            │   Feature Projection +   │
                            │       Gated Fusion       │
                            └────────────┬─────────────┘
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        ▼                                ▼                                ▼
┌───────────────┐                ┌───────────────┐                ┌───────────────┐
│ Classification│                │  Spatial Mask │                │ Temporal Mask │
│ Real / Fake / │                │ Manipulated   │                │ Manipulated   │
│ Physical Spoof│                │ pixels        │                │ frames        │
└───────────────┘                └───────────────┘                └───────────────┘
```

### Key Components
1. **DINOv3 ViT-B (Spatial Branch):** Captures static, per-frame visual anomalies such as edge blending artifacts and texture inconsistencies.
2. **V-JEPA 2.1 ViT-B (Temporal Branch):** Models multi-frame motion dynamics, blink naturalness, and facial temporal coherence.
3. **Adaptive Gated Fusion:** Dynamically balances spatial feature vector ($S$) and temporal feature vector ($T$) via a learned scalar gate $g \in [0, 1]$:
   $$\text{Fused Feature} = g \cdot S + (1 - g) \cdot T$$
4. **Multi-Task Output Heads:** Jointly computes 3-class classification probabilities (*Real*, *Deepfake*, *Physical Spoof*), spatial pixel manipulation masks ($H \times W$), and temporal frame manipulation masks.

---

## 🔄 Research Pipeline

The development workflow is structured into five distinct phases:

```
 [1. EDA] ──► [2. Preprocessing] ──► [3. Unimodal Baselines] ──► [4. Multimodal Fusion] ──► [5. Ablation & Error Analysis]
```

* **1. Exploratory Data Analysis (EDA):** Inspecting class balance, frame-rate consistency, video resolution, and face detector failure modes.
* **2. Preprocessing & Identity-Level Split:** Crop & sample 16 facial frames per video ($224 \times 224$). Splits (`train/val/test`) are enforced at the **identity level** to prevent data leakage.
* **3. Unimodal Baselines:** Training standalone DINOv3 and V-JEPA 2.1 models to establish baseline performance metrics.
* **4. Multimodal Fusion:** Training the complete STARD system using multi-task loss (Cross-Entropy + Spatial BCE + Temporal BCE).
* **5. Ablation Study:** Evaluating the contribution of Gated Fusion vs. Simple Concatenation, as well as the impact of Spatial/Temporal auxiliary masks.

---

## 📁 Repository Structure

```text
STARD/
├── README.md                   # Project overview & documentation
├── STARD_DEVELOPMENT_GUIDE.md  # Detailed architecture & code contract specifications
├── REPRODUCIBILITY.md          # Environment details & seed parameters
├── requirements.txt            # Python dependencies
│
├── datasets/                   # Dataset directory (git-ignored)
│   ├── raw/                    # Raw video files
│   ├── processed/              # Cropped 16-frame faces & CSV split files
│   └── cached_features/        # Pre-extracted backbone embeddings (.pt)
│
├── configs/                    # YAML configuration files
│   ├── base.yaml               # Shared hyperparameters & paths
│   ├── dinov3_baseline.yaml    # Unimodal DINOv3 config
│   ├── vjepa_baseline.yaml     # Unimodal V-JEPA config
│   ├── fusion_full.yaml        # Complete STARD system config
│   └── ablation/               # Configs for ablation experiments
│
├── notebooks/                  # Interactive notebooks
│   ├── 01_eda.ipynb            # Exploratory Data Analysis
│   └── 02_error_analysis.ipynb # Qualitative failure case analysis & mask visualization
│
├── src/                        # Core Python library
│   ├── contracts.py            # Shared data classes & tensor shape specifications
│   ├── data/                   # Data preprocessing, face tracking, & PyTorch Datasets
│   ├── models/                 # DINOv3, V-JEPA, Gated Fusion, & Multi-Task Heads
│   └── utils/                  # Multi-task loss functions & evaluation metrics (AUC, F1, IoU)
│
├── docs/                       # Internal specs (cached feature format, etc.)
│   └── cache_format.md         # Backbone embedding cache contract
│
├── scripts/                    # Entry-point execution scripts
│   ├── run_preprocess.py       # Face detection & identity split generation
│   ├── extract_features.py     # Offline backbone feature extraction
│   ├── train.py                # Main PyTorch training loop
│   ├── evaluate.py             # Model evaluation & benchmark exporter
│   └── smoke_dinov3.py         # Dataset-free smoke test for the spatial branch
│
└── outputs/                    # Experiment outputs (git-ignored)
    ├── checkpoints/            # Model weights (.pt)
    └── logs/                   # Training logs & TensorBoard summaries
```

---

## 🚀 Quickstart

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-org/STARD.git
cd STARD
pip install -r requirements.txt
```

### 2. Data Preprocessing

Extract 16 facial crops per video and generate identity-level dataset splits:

```bash
python scripts/run_preprocess.py --data_dir datasets/raw/ --output_dir datasets/processed/
```

### 3. Feature Extraction (Recommended)

Pre-extract embeddings from frozen DINOv3 and V-JEPA backbones to dramatically accelerate downstream training:

```bash
python scripts/extract_features.py --config configs/base.yaml
```

### 4. Training

Train unimodal baselines or the full STARD fusion model by passing the appropriate configuration file:

```bash
# Train DINOv3 Spatial Baseline
python scripts/train.py --config configs/dinov3_baseline.yaml

# Train V-JEPA Temporal Baseline
python scripts/train.py --config configs/vjepa_baseline.yaml

# Train Full STARD Model (Gated Fusion + Multi-Task Heads)
python scripts/train.py --config configs/fusion_full.yaml
```

### 5. Running Ablation Experiments

Run ablation experiments by supplying configs from the `configs/ablation/` directory:

```bash
# Evaluate Simple Concatenation vs. Gated Fusion
python scripts/train.py --config configs/ablation/concat_fusion.yaml
```

---

## 📊 Results & Benchmarks

*Detailed benchmarking tables for FaceForensics++, Celeb-DF, and custom physical spoof datasets will be populated upon experiment completion.*

| Model Variant | Fusion Type | Multi-Task Masking | Cls AUC (%) | Spatial IoU | Temporal F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| DINOv3 (Spatial Only) | N/A | ❌ | - | - | - |
| V-JEPA 2.1 (Temporal Only) | N/A | ❌ | - | - | - |
| STARD (Concat) | Concatenation | ❌ | - | - | - |
| **STARD (Full)** | **Gated Fusion** | **✅** | **TBD** | **TBD** | **TBD** |

---

## 📖 Technical Documentation

For in-depth explanations of boundary contracts (`contracts.py`), tensor shape requirements, and team module integration, refer to the [Development Guide](STARD_DEVELOPMENT_GUIDE.md).

---

## 📜 License

This repository is released under the [MIT License](LICENSE).

> **SUPERSEDED — kept for history only.**
> The project moved to **STAF-Net** (binary DFDC detection, DINOv3 ViT-S/16 +
> V-JEPA 2.1 ViT-B/16, frozen-vs-LoRA ablations, no mask heads). The active
> contract is `src/contracts.py` v0.2 and the active plan lives in
> `docs/open_decisions.md` + `README.md`. Everything below describes the
> earlier 3-class + spatial/temporal-mask design and no longer matches the code.

# STARD: Spatial-Temporal Deepfake Detection System
> **Panduan Arsitektur, Kontrak Modul, dan Workflow Pengembangan Tim**

Dokumen ini berisi dokumentasi teknis komprehensif untuk arsitektur *Deepfake Detection* berbasis dual-backbone (**DINOv3** + **V-JEPA 2.1**), pembagian kerja tim 3 orang, struktur repositori, serta spesifikasi kode dan kontrak antarmuka (*boundary contract*).

---

## 1. Arsitektur Sistem

Sistem ini dirancang untuk mendeteksi manipulasi video (baik *Deepfake* generatif maupun *Physical Spoofing*) dengan menggabungkan bukti spatial (visual statis per-frame) dan bukti temporal (gerakan antar-frame).

```
                            +--------------------------+
                            |       Input Video        |
                            | 16 frame, facial crops   |
                            +------------+-------------+
                                         |
                   +---------------------+---------------------+
                   |                                           |
                   v                                           v
    +------------------------------+           +------------------------------+
    |        DINOv3 ViT-B          |           |       V-JEPA 2.1 ViT-B       |
    |          (Spatial)           |           |          (Temporal)          |
    | Per-frame patch embeddings,  |           | Video embeddings, motion     |
    | blending boundaries, local   |           | context, temporal            |
    | artifacts                    |           | correspondence               |
    +--------------+---------------+           +--------------+---------------+
                   |                                           |
                   +---------------------+---------------------+
                                         |
                                         v
                            +--------------------------+
                            |   Feature Projection +   |
                            |       Gated Fusion       |
                            +------------+-------------+
                                         |
        +--------------------------------+--------------------------------+
        |                                |                                |
        v                                v                                v
+---------------+                +---------------+                +---------------+
| Classification|                |  Spatial Mask |                | Temporal Mask |
| Real / Fake / |                | Manipulated   |                | Manipulated   |
| Physical Spoof|                | pixels        |                | frames        |
+---------------+                +---------------+                +---------------+
```

### 1.1 Komponen Otak & Sub-Network
Total arsitektur ini terdiri dari **1 model terintegrasi** dengan **6 *sub-network***:
1. **DINOv3 ViT-B Backbone (Spatial):** Mengekstrak keanehan visual piksel lokal per-frame (misal: garis batas tempelan wajah atau *artifacts* tekstur).
2. **V-JEPA 2.1 ViT-B Backbone (Temporal):** Mengekstrak fitur gerakan 3D dan konsistensi pergerakan wajah antar-frame.
3. **Feature Projection + Gated Fusion Module:** Menyamakan dimensi fitur dan menggabungkan informasi spatial ($S$) dan temporal ($T$) secara adaptif:
   $$\text{Fitur Gabungan} = g \cdot S + (1 - g) \cdot T$$
   di mana $g \in [0, 1]$ adalah nilai gerbang (*gate*) yang dihitung dinamis menggunakan fungsi aktivasi Sigmoid.
4. **Classification Head:** Memprediksi probabilitas kelas 3 arah (*Real*, *Deepfake*, atau *Physical Spoof*).
5. **Spatial Mask Head:** Memprediksi lokasi piksel yang dimanipulasi pada setiap frame ($H \times W$).
6. **Temporal Mask Head:** Memprediksi indeks frame mana saja yang mengalami manipulasi.

---

## 2. Pipeline Eksperimen

Pengembangan dilakukan melalui 5 tahapan terstruktur:

1. **EDA (Exploratory Data Analysis):**
   - Analisis distribusi kelas (*Real*, *Deepfake*, *Physical Spoof*).
   - Cek variasi FPS, resolusi, durasi, dan *failure rate* pada deteksi wajah (*face detector/tracker*).
2. **Preprocessing:**
   - Ekstraksi 16 frame ($16 \text{ frames}$) per video (*consecutive* atau *uniform sampling*).
   - Crop area wajah dan resize ke resolusi target ($224 \times 224$).
   - Pembagian data (*Train / Val / Test*) secara **Identity-level split** (mencegah *leakage* identitas wajah yang sama di train dan test).
3. **Modelling (Unimodal Baselines):**
   - **Orang A:** Melatih baseline **DINOv3 + Heads** (Spatial saja).
   - **Orang B:** Melatih baseline **V-JEPA 2.1 + Heads** (Temporal saja).
4. **Fusion (Multimodal Integration):**
   - **Orang C / Tim:** Menggabungkan output dari kedua backbone menggunakan *Feature Projection + Gated Fusion* dan melatih *multi-task loss*.
5. **Ablation Study & Error Analysis:**
   - Membandingkan *Gated Fusion* vs *Simple Concatenation*.
   - Menguji efek ada/tidaknya *Spatial Mask* dan *Temporal Mask* terhadap akurasi klasifikasi.
   - Analisis kasus salah prediksi (*false positive* / *false negative*).

---

## 3. Pembagian Kerja Tim (3 Roles)

| Anggota Tim | Tanggung Jawab Utama | Target File / Module |
| :--- | :--- | :--- |
| **Orang 1 (Data & Preproc Lead)** | EDA, Video Loading, Face Tracking, Sampling 16 frame, PyTorch Dataset, Identity Splitting. | `src/data/preprocess.py`<br>`src/data/dataset.py`<br>`notebooks/01_eda.ipynb` |
| **Orang A (Spatial Model Lead)** | Wrapper DINOv3 ViT-B, Normalisasi Spatial, Feature Extraction Spatial, Unimodal Baseline DINOv3. | `src/models/dinov3.py`<br>`configs/dinov3_baseline.yaml` |
| **Orang B (Temporal Model Lead)** | Wrapper V-JEPA 2.1 ViT-B, Alignment Token Waktu, Feature Extraction Temporal, Unimodal Baseline V-JEPA. | `src/models/vjepa.py`<br>`configs/vjepa_baseline.yaml` |
| **Integrasi & Training (Orang C / Shared)** | Gated Fusion, Multi-task Heads, Multi-task Loss, Training Loop, Script Evaluasi & Ablation. | `src/contracts.py`<br>`src/models/fusion.py`<br>`src/models/heads.py`<br>`scripts/train.py` |

---

## 4. Boundary Contracts (`src/contracts.py`)

Kunci kerja paralel tanpa saling menunggu adalah **menyepakati tipe data dan dimensi tensor** di awal.

```python
"""
src/contracts.py
File kontrak utama yang disepakati oleh seluruh tim.
"""

from dataclasses import dataclass
import torch

@dataclass
class BatchInput:
    """Input batch dari DataLoader (Orang 1)."""
    frames: torch.Tensor         # Shape: (B, 16, 3, 224, 224) - Tensor Float32 [0.0, 1.0]
    label_cls: torch.Tensor      # Shape: (B,) - LongTensor: 0 (Real), 1 (Deepfake), 2 (Spoof)
    label_spatial: torch.Tensor  # Shape: (B, 16, 224, 224) - FloatTensor 0.0/1.0
    label_temporal: torch.Tensor # Shape: (B, 16) - FloatTensor 0.0/1.0

@dataclass
class BackboneFeatures:
    """Output fitur dari DINOv3 (Orang A) dan V-JEPA 2.1 (Orang B)."""
    spatial: torch.Tensor        # Shape: (B, 16, P, D_spatial)  e.g., (B, 16, 196, 768)
    temporal: torch.Tensor       # Shape: (B, 16, P, D_temporal) e.g., (B, 16, 196, 768)

@dataclass
class ModelOutputs:
    """Output prediksi dari MultiTaskHeads."""
    logits_cls: torch.Tensor     # Shape: (B, 3) - Raw logits klasifikasi
    mask_spatial: torch.Tensor   # Shape: (B, 16, 224, 224) - Prediksi piksel terinfeksi
    mask_temporal: torch.Tensor  # Shape: (B, 16) - Prediksi frame terinfeksi
```

---

## 5. Struktur Repositori Projek

```text
STARD/
├── README.md
├── REPRODUCIBILITY.md
├── STARD_DEVELOPMENT_GUIDE.md # Document ini
├── requirements.txt
├── .gitignore
├── assets/                     # Diagram arsitektur & grafik visualisasi
├── paper/                      # Source code LaTeX paper
│
├── datasets/                   # [Di-ignore Git]
│   ├── raw/                    # Video asal
│   ├── processed/              # Face crops 16 frame & split index (.csv)
│   └── cached_features/        # Saved embeddings (.pt) hasil ekstraksi DINO & V-JEPA
│
├── configs/                    # File konfigurasi YAML eksperimen
│   ├── base.yaml               # Hyperparameter umum & path
│   ├── dinov3_baseline.yaml    # Config baseline Spatial
│   ├── vjepa_baseline.yaml     # Config baseline Temporal
│   ├── fusion_full.yaml        # Config arsitektur lengkap
│   └── ablation/               # Variasi eksperimen (concat vs gated, no_mask, dll.)
│
├── notebooks/                  # Notebook eksploratif
│   ├── 01_eda.ipynb            # Analisis awal data
│   └── 02_error_analysis.ipynb # Visualisasi kesalahan prediksi & mask output
│
├── src/                        # Modul Python utama
│   ├── __init__.py
│   ├── contracts.py            # Dataclass kontrak tensor
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocess.py       # Face tracking, cropping & uniform sampling
│   │   ├── dataset.py          # PyTorch Dataset (raw frames & cached features)
│   │   └── transforms.py       # Data augmentasi video/image
│   ├── models/
│   │   ├── __init__.py
│   │   ├── dinov3.py           # Backbone DINOv3 ViT-B (Orang A)
│   │   ├── vjepa.py            # Backbone V-JEPA 2.1 ViT-B (Orang B)
│   │   ├── fusion.py           # Feature Projection & Gated Fusion (Orang C)
│   │   ├── heads.py            # Classification, Spatial & Temporal heads (Orang C)
│   │   └── detector.py         # Full Pipeline Assembler
│   └── utils/
│       ├── __init__.py
│       ├── losses.py           # Combined Multi-Task Loss Function
│       └── metrics.py          # AUC, F1-Score, dan Mask IoU Evaluator
│
├── scripts/                    # Entry points eksekusi CLI
│   ├── run_preprocess.py       # Menjalankan ekstraksi wajah & pembuat split
│   ├── extract_features.py     # Caching embeddings dari backbone ke disk
│   ├── train.py                # Script training terpadu berbasis YAML config
│   └── evaluate.py             # Script pengujian model pada Test set
│
└── outputs/                    # [Di-ignore Git]
    ├── checkpoints/            # File bobot model (.pt / .pth)
    └── logs/                   # TensorBoard / WandB logs
```

---

## 6. Template Implementasi Kode Inti

### 6.1 `src/models/dinov3.py` (Orang A)
```python
import torch
import torch.nn as nn
import torchvision.transforms as T
from src.contracts import BackboneFeatures

class DINOv3SpatialExtractor(nn.Module):
    def __init__(self, pretrained_name="dinov3_vitb14"):
        super().__init__()
        # Load pre-trained DINOv3 model
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14') # Proxy / update URL DINOv3
        
        # Freeze weights
        for p in self.parameters():
            p.requires_grad = False
            
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames shape: (B, 16, 3, 224, 224)
        B, T_len, C, H, W = frames.shape
        flat_frames = frames.view(B * T_len, C, H, W)
        flat_norm = self.normalize(flat_frames)
        
        # Extracted patch tokens
        features = self.backbone.forward_features(flat_norm)["x_norm_patchtokens"]
        _, P, D_s = features.shape  # P = 196 patch, D_s = 768
        
        spatial_features = features.view(B, T_len, P, D_s)
        return spatial_features
```

### 6.2 `src/models/vjepa.py` (Orang B)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VJEPATemporalExtractor(nn.Module):
    def __init__(self, pretrained_name="vjepa_2.1_vitb"):
        super().__init__()
        # Load pre-trained V-JEPA 2.1 model
        self.backbone = torch.hub.load('facebookresearch/vjepa', 'vjepa_vitb') # Proxy / update URL V-JEPA 2.1
        
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames shape: (B, 16, 3, 224, 224)
        B, T_len, C, H, W = frames.shape
        
        # Reorder dimensi untuk 3D Conv / Video Transformer jika diperlukan: (B, C, T, H, W)
        video_input = frames.permute(0, 2, 1, 3, 4)
        
        # Run backbone
        out_features = self.backbone(video_input) # e.g. Output token (B, T_out, P, D_t)
        
        # Alignment temporal jika T_out != 16 (misal V-JEPA memakai tubelet stride 2 -> T_out = 8)
        if out_features.shape[1] != T_len:
            out_features = out_features.permute(0, 2, 3, 1) # (B, P, D_t, T_out)
            out_features = F.interpolate(out_features, size=T_len, mode='linear', align_corners=False)
            out_features = out_features.permute(0, 3, 1, 2) # (B, 16, P, D_t)
            
        return out_features
```

### 6.3 `src/models/fusion.py` (Orang C / Shared)
```python
import torch
import torch.nn as nn
from src.contracts import BackboneFeatures

class GatedFusionModule(nn.Module):
    def __init__(self, d_spatial=768, d_temporal=768, d_hidden=512):
        super().__init__()
        # 1. Feature Projection
        self.proj_spatial = nn.Linear(d_spatial, d_hidden)
        self.proj_temporal = nn.Linear(d_temporal, d_hidden)
        
        # 2. Gated Mechanism
        self.gate_network = nn.Sequential(
            nn.Linear(d_hidden * 2, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.Sigmoid()
        )

    def forward(self, feats: BackboneFeatures) -> torch.Tensor:
        s_proj = self.proj_spatial(feats.spatial)   # (B, 16, P, d_hidden)
        t_proj = self.proj_temporal(feats.temporal) # (B, 16, P, d_hidden)
        
        # Concatenate untuk input ke gate network
        concat_feats = torch.cat([s_proj, t_proj], dim=-1) # (B, 16, P, 2 * d_hidden)
        
        # Hitung bobot gate g in [0, 1]
        g = self.gate_network(concat_feats)
        
        # Adaptif Convex Combination
        fused_features = g * s_proj + (1.0 - g) * t_proj
        return fused_features # Shape: (B, 16, P, d_hidden)
```

### 6.4 `src/models/heads.py` (Orang C / Shared)
```python
import torch
import torch.nn as nn
from src.contracts import ModelOutputs

class MultiTaskHeads(nn.Module):
    def __init__(self, d_hidden=512, num_classes=3):
        super().__init__()
        # 1. Classification Head (Real / Deepfake / Spoof)
        self.cls_head = nn.Sequential(
            nn.Linear(d_hidden, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        
        # 2. Temporal Mask Head (Per-frame prediction)
        self.temporal_head = nn.Linear(d_hidden, 1)
        
        # 3. Spatial Mask Head (Upsample patch features to 224x224 mask)
        self.spatial_upsample = nn.Sequential(
            nn.ConvTranspose2d(d_hidden, 256, kernel_size=4, stride=2, padding=1), # 14x14 -> 28x28
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),  # 28x28 -> 56x56
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 16, kernel_size=4, stride=2, padding=1),  # 56x56 -> 112x112
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),   # 112x112 -> 224x224
        )

    def forward(self, fused_tokens: torch.Tensor) -> ModelOutputs:
        # fused_tokens shape: (B, 16, P, D) di mana P = 196 (14x14 patch)
        B, T_len, P, D = fused_tokens.shape
        
        # Global Pooling untuk Klasifikasi
        pooled_cls = fused_tokens.mean(dim=(1, 2)) # (B, D)
        logits_cls = self.cls_head(pooled_cls)     # (B, 3)
        
        # Pooling Patch untuk Temporal Mask
        pooled_temporal = fused_tokens.mean(dim=2) # (B, 16, D)
        logits_temporal = self.temporal_head(pooled_temporal).squeeze(-1) # (B, 16)
        
        # Reshape Patch ke 2D Spatial Grid untuk Spatial Mask
        grid_dim = int(P ** 0.5) # 14
        spatial_in = fused_tokens.view(B * T_len, grid_dim, grid_dim, D).permute(0, 3, 1, 2) # (B*16, D, 14, 14)
        logits_spatial = self.spatial_upsample(spatial_in).view(B, T_len, 224, 224)         # (B, 16, 224, 224)
        
        return ModelOutputs(
            logits_cls=logits_cls,
            mask_spatial=logits_spatial,
            mask_temporal=logits_temporal
        )
```

---

## 7. Skrip Inisialisasi & Workflow Eksekusi

### 7.1 Command Terminal untuk Setup Folder
Jalankan perintah berikut di folder *root* proyek (`~/arun/STARD`):

```bash
# Hapus folder code kosongan jika ada
[ -d "code" ] && rmdir code

# Buat hirarki folder standar
mkdir -p datasets/{raw,processed,cached_features} \
         configs/ablation \
         notebooks \
         src/{data,models,utils} \
         scripts \
         outputs/{checkpoints,logs}

# Buat file __init__.py dan contracts.py
touch src/__init__.py \
      src/data/__init__.py \
      src/models/__init__.py \
      src/utils/__init__.py \
      src/contracts.py
```

### 7.2 Konfigurasi `.gitignore`
Pastikan file `.gitignore` berisi entri berikut agar file besar tidak membebani repositori Git:

```gitignore
# Data & Outputs
datasets/
outputs/
*.pt
*.pth
*.ckpt
*.npy
*.h5

# Python & Environment
__pycache__/
*.py[cod]
.ipynb_checkpoints/
.env
venv/
```

### 7.3 Trik Penghematan VRAM & Kecepatan Training (*Feature Caching*)
Menjalankan DINOv3 dan V-JEPA 2.1 secara berulang pada setiap *epoch* akan memakan memori GPU yang sangat besar dan memperlambat proses *training*. 

Karena kedua backbone dibekukan (*frozen weights*), gunakan alur kerja berikut:
1. Menjalankan `scripts/extract_features.py` **sekali saja** untuk mengekstrak vektor fitur dari dataset. Simpan file `.pt` di `datasets/cached_features/`.
2. Saat melatih *Fusion Module* dan *Heads* (pada `scripts/train.py`), `DataLoader` cukup membaca file `.pt` dari disk. Hal ini memungkinkan training dilakukan puluhan kali lipat lebih cepat dengan VRAM GPU yang minimal.

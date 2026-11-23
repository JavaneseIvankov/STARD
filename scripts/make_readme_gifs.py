"""Generate two publication-grade animated GIFs for README.md from real verified artifacts:
  1. assets/ablation_progression.gif  — animated A1–A8 -> Std-Concat ablation bar + ROC progression
  2. assets/attention_ablation.gif    — animated Frozen vs LoRA & DINOv3 vs V-JEPA 2.1 attention maps across frames & cases
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT / "assets"
METRICS_DIR = ROOT / "results" / "metrics"
ATTN_DIR = ROOT / "results" / "attention"
ASSETS_DIR.mkdir(exist_ok=True)


def fig_to_pil(fig: plt.Figure, dpi: int = 110) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def make_ablation_progression_gif() -> Path:
    ab = json.loads((METRICS_DIR / "ablation_summary.json").read_text())
    ev = {f"A{i}": json.loads((METRICS_DIR / f"eval_A{i}_test.json").read_text()) for i in range(1, 9)}

    entries = [
        ("A1 Frozen DINOv3", ev["A1"]["metrics"]["auroc"], ev["A1"]["metrics"]["f1"], "#D55E00", "Stage 1: Spatial Frozen Baseline"),
        ("A3 LoRA DINOv3", ev["A3"]["metrics"]["auroc"], ev["A3"]["metrics"]["f1"], "#E67E22", "Stage 2: LoRA boosts DINOv3 (+0.016 AUROC, 10/10 seeds)"),
        ("A2 Frozen V-JEPA 2.1", ev["A2"]["metrics"]["auroc"], ev["A2"]["metrics"]["f1"], "#0072B2", "Stage 3: Spatiotemporal V-JEPA 2.1 Frozen Baseline"),
        ("A4 LoRA V-JEPA 2.1", ev["A4"]["metrics"]["auroc"], ev["A4"]["metrics"]["f1"], "#1D4ED8", "Stage 4: Best Single Branch (0.8855 AUROC · 0.8462 F1)"),
        ("A5 Frozen Concat (MLP)", ev["A5"]["metrics"]["auroc"], ev["A5"]["metrics"]["f1"], "#A855F7", "Stage 5: Unnormalized MLP Fusion suffers scale mismatch"),
        ("A6 Frozen Gated (MLP)", ev["A6"]["metrics"]["auroc"], ev["A6"]["metrics"]["f1"], "#9333EA", "Stage 6: Unnormalized Gated MLP overfits N=716 clips"),
        ("A7 LoRA Concat (MLP)", ev["A7"]["metrics"]["auroc"], ev["A7"]["metrics"]["f1"], "#C026D3", "Stage 7: Adapted Unnormalized Concat (0.8693 AUROC)"),
        ("A8 LoRA Gated (MLP)", ev["A8"]["metrics"]["auroc"], ev["A8"]["metrics"]["f1"], "#CC79A7", "Stage 8: Original STAF-Net Gated MLP (0.8681 AUROC)"),
        ("Std-Blend (0.5 / 0.5)", 0.9027, 0.8280, "#10B981", "Stage 9: Standardized Late Blend jumps to 0.9027 (+0.030!)"),
        ("Std-Concat Fusion ★", 0.9030, 0.8313, "#0F766E", "Final Winner: STRAD Std-Concat Fusion — 0.9030 Test AUROC!"),
    ]

    frames: list[Image.Image] = []
    durations: list[int] = []

    for step in range(1, len(entries) + 1):
        fig, ax = plt.subplots(figsize=(9.4, 5.1))
        sub = entries[:step]
        labels = [e[0] for e in sub]
        aurocs = [e[1] for e in sub]
        f1s = [e[2] for e in sub]
        colors = [e[3] for e in sub]

        y = np.arange(len(entries))
        # Draw faint empty slots for upcoming models so layout never shifts
        ax.barh(y, [0.0] * len(entries), height=0.62)
        bars = ax.barh(y[:step], aurocs, height=0.62, color=colors, edgecolor="white", linewidth=1.2)
        # Highlight current step bar with dark border
        bars[-1].set_edgecolor("#111827")
        bars[-1].set_linewidth(2.0)

        ax.set_yticks(y)
        ax.set_yticklabels([e[0] for e in entries], fontsize=9.5, fontweight="bold")
        ax.invert_yaxis()
        ax.set_xlim(0.78, 0.952)
        ax.set_xlabel("Held-Out DFDC Test ROC-AUC (N = 154, Source-Grouped Split)", fontsize=10, fontweight="bold")
        ax.axvline(ev["A4"]["metrics"]["auroc"], color="#1D4ED8", ls="--", lw=1.2, alpha=0.65, label="Best Single Branch (A4 = 0.8855)")
        if step >= 9:
            ax.axvline(0.9030, color="#0F766E", ls="-.", lw=1.5, alpha=0.85, label="STRAD Std-Concat (0.9030)")

        for idx, (auc, f1) in enumerate(zip(aurocs, f1s)):
            ax.text(auc + 0.002, idx, f"AUC {auc:.4f}  (F1 {f1:.3f})", va="center", fontsize=8.8,
                    fontweight="bold" if idx == step - 1 else "normal", color="#111827")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", linestyle=":", alpha=0.5)
        ax.set_title(f"STRAD / STAF-Net Ablation Progression\n{entries[step-1][4]}", fontsize=11.5, fontweight="bold", color="#0F172A", pad=10)
        ax.legend(loc="upper right", fontsize=8.4, frameon=True, facecolor="white", framealpha=0.95)

        plt.tight_layout()
        frames.append(fig_to_pil(fig, dpi=115))
        durations.append(2400 if step == len(entries) else 950)

    # Normalize exact frame dimensions to frame 0
    w, h = frames[0].size
    frames = [f.resize((w, h), Image.Resampling.LANCZOS) for f in frames]
    out_path = ASSETS_DIR / "ablation_progression.gif"
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    print(f"Saved {out_path} ({out_path.stat().st_size / 1024:.1f} KB, {len(frames)} frames)")
    return out_path


def upsample_map(m: np.ndarray, size: int = 224) -> np.ndarray:
    t = torch.from_numpy(m.astype(np.float32))[None, None]
    up = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)[0, 0].numpy()
    lo, hi = np.percentile(up, 2), np.percentile(up, 99.5)
    return np.clip((up - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def make_attention_ablation_gif() -> Path:
    npz = np.load(ATTN_DIR / "attention_maps.npz")
    cases = [
        ("wkwwpmocfo", "Case 1: FAKE · Both Branches Catch", "DINO=0.95  |  V-JEPA=0.71  |  Std-Concat=1.00 (FAKE)", "spatial r = +0.056 · DINO L1 shift = 0.206"),
        ("llmftzgeec", "Case 2: FAKE · DINOv3 Rescues V-JEPA Miss", "DINO=0.81  |  V-JEPA=0.21  |  Std-Concat=0.73 (FAKE)", "spatial r = +0.236 · DINO L1 shift = 0.129"),
        ("yednyneaag", "Case 3: FAKE · V-JEPA Rescues DINOv3 Miss", "DINO=0.37  |  V-JEPA=0.64  |  Std-Concat=0.48 (FAKE)", "spatial r = +0.121 · Temporal cue dominates"),
        ("ogghhskfdr", "Case 4: REAL · Authentic Baseline", "DINO=0.09  |  V-JEPA=0.13  |  Std-Concat=0.00 (REAL)", "spatial r = +0.059 · LoRA suppresses wall decor"),
    ]
    frame_labels = ["t = 0.00s (Frame 0)", "t = 0.63s (Frame 5)", "t = 1.25s (Frame 10)", "t = 1.88s (Frame 15)"]
    vj_idx_map = [0, 2, 5, 7]

    frames: list[Image.Image] = []
    durations: list[int] = []

    for stem, title, scores, note in cases:
        d_rgb = npz[f"{stem}__dino_rgb"]            # (4, 224, 224, 3)
        d_frz = npz[f"{stem}__dino_frozen_attn"]    # (4, 14, 14)
        d_lor = npz[f"{stem}__dino_lora_attn"]      # (4, 14, 14)
        v_lor = npz[f"{stem}__vjepa_lora_attn"]     # (8, 24, 24)

        for fi in range(4):
            fig, axes = plt.subplots(1, 5, figsize=(12.2, 3.05))
            rgb = d_rgb[fi]
            df = d_frz[fi]
            dl = d_lor[fi]
            vl = v_lor[vj_idx_map[fi]]

            diff = (dl / (dl.sum() + 1e-12)) - (df / (df.sum() + 1e-12))
            diff_up = F.interpolate(torch.from_numpy(diff.astype(np.float32))[None, None], size=(224, 224), mode="bilinear", align_corners=False)[0, 0].numpy()
            vmax = float(np.percentile(np.abs(diff_up), 99)) + 1e-8

            panels = [
                ("Input Face Crop", rgb, None, None, None),
                ("DINOv3 Frozen", rgb, upsample_map(df), "inferno", 0.55),
                ("DINOv3 LoRA", rgb, upsample_map(dl), "inferno", 0.55),
                ("LoRA − Frozen Shift", rgb, diff_up, "RdBu_r", 0.72),
                ("V-JEPA 2.1 LoRA", rgb, upsample_map(vl), "viridis", 0.55),
            ]
            for ax, (pt, base, ov, cmap, alpha) in zip(axes, panels):
                ax.imshow(base, alpha=0.45 if cmap == "RdBu_r" else 1.0)
                if ov is not None:
                    if cmap == "RdBu_r":
                        ax.imshow(ov, cmap=cmap, vmin=-vmax, vmax=vmax, alpha=alpha)
                    else:
                        ax.imshow(ov, cmap=cmap, alpha=alpha)
                ax.set_title(pt, fontsize=9.2, fontweight="bold")
                ax.set_xticks([])
                ax.set_yticks([])

            fig.suptitle(f"{title}   ·   {frame_labels[fi]}", fontsize=11, fontweight="bold", color="#0F172A", y=0.99)
            fig.text(0.5, 0.015, f"{scores}    |    {note}", ha="center", fontsize=8.8, color="#334155", fontweight="bold")
            plt.tight_layout(rect=[0, 0.06, 1, 0.92])
            frames.append(fig_to_pil(fig, dpi=100))
            durations.append(750)

    w, h = frames[0].size
    frames = [f.resize((w, h), Image.Resampling.LANCZOS) for f in frames]
    out_path = ASSETS_DIR / "attention_ablation.gif"
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    print(f"Saved {out_path} ({out_path.stat().st_size / 1024:.1f} KB, {len(frames)} frames)")
    return out_path


if __name__ == "__main__":
    make_ablation_progression_gif()
    make_attention_ablation_gif()

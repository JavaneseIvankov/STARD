"""Extract real last-block attention maps from DINOv3 ViT-S/16 and V-JEPA 2.1 ViT-B/16.

Runs both Frozen and Stage-I LoRA checkpoints on selected DFDC test-set clips and
saves a compact compressed NPZ (`results/attention/attention_maps.npz`) containing:
- raw RGB face frames (4 DINO frames at 224x224 + 8 representative V-JEPA frames at 384x384)
- DINOv3 last-block CLS->patch attention maps (4, 14, 14) for frozen & lora
- DINOv3 per-head CLS->patch attention maps (6, 14, 14) for frame 0 (lora)
- V-JEPA 2.1 last-block received spatiotemporal attention maps (8, 24, 24) for frozen & lora
- V-JEPA 2.1 temporal attention energy (8,) and spatial-mean attention (24, 24)
- Summary attention metrics (entropy, center-face concentration ratio, Frozen->LoRA L1 shift)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.data.video_dataset import ClipDataset, read_manifest
from src.models.dino_branch import DinoBranch
from src.models.vjepa_branch import VJepaBranch


DEFAULT_CLIPS = [
    ("llmftzgeec.mp4", "FAKE_DINO_WINS"),
    ("mxgdaffjmt.mp4", "FAKE_DINO_WINS_2"),
    ("yednyneaag.mp4", "FAKE_VJEPA_WINS"),
    ("hkelfxsxyo.mp4", "FAKE_VJEPA_WINS_2"),
    ("wkwwpmocfo.mp4", "FAKE_BOTH_CATCH"),
    ("ogghhskfdr.mp4", "REAL_BOTH_CONFIDENT"),
    ("jrofvutpmh.mp4", "FAKE_HARD_FN"),
]


def _unwrap_backbone(backbone: torch.nn.Module) -> torch.nn.Module:
    if hasattr(backbone, "get_base_model"):
        return backbone.get_base_model()
    return backbone


@torch.no_grad()
def extract_dino_attention(
    model: DinoBranch,
    dino_frames: torch.Tensor,  # (1, 4, 3, 224, 224) float [0, 1]
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (cls_to_patch (4, 14, 14), per_head_f0 (H, 14, 14), logit)."""
    base = _unwrap_backbone(model.backbone)
    last_attn = base.blocks[-1].attn
    orig_fused = getattr(last_attn, "fused_attn", True)
    last_attn.fused_attn = False

    captured: list[torch.Tensor] = []

    def _hook(_mod, _inp, out):
        # attn_drop receives post-softmax attention (B_flat, H, N, N)
        captured.append(out.detach().float().cpu())

    handle = last_attn.attn_drop.register_forward_hook(_hook)
    try:
        x = dino_frames.to(device)
        logit = float(model.predict_logit(x).squeeze().item())
    finally:
        handle.remove()
        last_attn.fused_attn = orig_fused

    attn = captured[0]  # (4, H, N, N), N = 1 + 4 reg + 196 patches = 201
    n_tokens = attn.shape[-1]
    n_patches = 14 * 14
    prefix = n_tokens - n_patches  # 5 (1 CLS + 4 registers)
    cls_to_patch = attn[:, :, 0, prefix:]  # (4, H, 196)
    cls_mean = cls_to_patch.mean(dim=1).reshape(4, 14, 14).numpy()
    per_head_f0 = cls_to_patch[0].reshape(-1, 14, 14).numpy()
    return cls_mean, per_head_f0, logit


@torch.no_grad()
def extract_vjepa_attention(
    model: VJepaBranch,
    vjepa_frames: torch.Tensor,  # (1, 3, 16, 384, 384) float [0, 1]
    device: torch.device,
) -> tuple[np.ndarray, float]:
    """Return (received_attn (8, 24, 24), logit) from the last V-JEPA 2.1 encoder block."""
    base = _unwrap_backbone(model.backbone)
    last_layer = base.encoder.layer[-1]
    orig_attn_impl = getattr(base.config, "_attn_implementation", None)

    # Monkey-patch last_layer.attention.forward to compute eager attention weights
    # (only on the final layer so memory stays tiny: 1 x 12 x 4608 x 4608 in fp16 ~500 MB)
    attn_mod = last_layer.attention
    orig_forward = attn_mod.forward
    captured: list[torch.Tensor] = []

    def _eager_capture(hidden_states, position_mask=None, T=None, H_patches=None, W_patches=None, output_attentions=False):
        # Compute eager attention using the module's own projections & RoPE
        B, N, C = hidden_states.shape
        q = attn_mod.query(hidden_states).view(B, N, attn_mod.num_attention_heads, attn_mod.attention_head_size).transpose(1, 2)
        k = attn_mod.key(hidden_states).view(B, N, attn_mod.num_attention_heads, attn_mod.attention_head_size).transpose(1, 2)
        if hasattr(attn_mod, "apply_rotary_embeddings"):
            if position_mask is not None:
                ids = position_mask.unsqueeze(1).repeat(1, attn_mod.num_attention_heads, 1)
            else:
                ids = torch.arange(N, device=hidden_states.device)
            q = attn_mod.apply_rotary_embeddings(q, ids, T=T, H_patches=H_patches, W_patches=W_patches)
            k = attn_mod.apply_rotary_embeddings(k, ids, T=T, H_patches=H_patches, W_patches=W_patches)
        # Compute softmax attention in float16/bf16 on GPU, then immediately reduce over query & head dims
        scale = attn_mod.attention_head_size ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (1, H, N, N)
        probs = torch.softmax(scores.float(), dim=-1)
        # Received attention mass per token j: mean over heads and all query tokens i
        received = probs.mean(dim=(1, 2)).squeeze(0).cpu()  # (N,)
        captured.append(received)
        return orig_forward(hidden_states, position_mask=position_mask, T=T, H_patches=H_patches, W_patches=W_patches, output_attentions=output_attentions)

    attn_mod.forward = _eager_capture
    try:
        x = vjepa_frames.to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logit = float(model.predict_logit(x).squeeze().item())
    finally:
        attn_mod.forward = orig_forward
        if orig_attn_impl is not None:
            base.config._attn_implementation = orig_attn_impl

    received = captured[0].numpy()  # (4608,) = 8 * 24 * 24
    t_grid = received.size // (24 * 24)
    return received.reshape(t_grid, 24, 24), logit


def attention_stats(amap: np.ndarray) -> dict[str, float]:
    """Compute normalized entropy and central-face attention ratio on a (H, W) map."""
    p = np.clip(amap.astype(np.float64), 0.0, None)
    p = p / (p.sum() + 1e-12)
    h, w = p.shape
    ent = float(-np.sum(p * np.log(p + 1e-12)) / np.log(h * w))
    r0, r1 = int(h * 0.2), int(h * 0.8)
    c0, c1 = int(w * 0.2), int(w * 0.8)
    center_ratio = float(p[r0:r1, c0:c1].sum())
    peak_to_mean = float(p.max() / (p.mean() + 1e-12))
    return {"norm_entropy": ent, "center_ratio": center_ratio, "peak_to_mean": peak_to_mean}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/manifest.csv")
    parser.add_argument("--clips-dir", default="data/clips")
    parser.add_argument("--dino-weights", default="weights/dinov3-vits16")
    parser.add_argument("--vjepa-weights", default="weights/vjepa2.1-vitb16")
    parser.add_argument("--dino-ckpt", default="checkpoints/dino_lora.pt")
    parser.add_argument("--vjepa-ckpt", default="checkpoints/vjepa_lora.pt")
    parser.add_argument("--out-dir", default="results/attention")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_list = read_manifest(args.manifest)
    rows = {r.video_id: r for r in rows_list}
    selected = [(vid, tag) for vid, tag in DEFAULT_CLIPS if vid in rows]
    print(f"Extracting attention maps for {len(selected)} test clips on {device}...")

    sel_rows = [rows[vid] for vid, _ in selected]
    ds = ClipDataset(sel_rows, args.clips_dir, train=False, emit_dino=True, emit_vjepa=True)

    # Load all clips into memory using the exact Stage I dataset pipeline
    clips_data = {}
    for idx, (vid, tag) in enumerate(selected):
        r = rows[vid]
        item = ds[idx]
        dino_t = item["dino_frames"].unsqueeze(0)    # (1, 4, 3, 224, 224) float [0, 1]
        vjepa_t = item["vjepa_frames"].unsqueeze(0)  # (1, 3, 16, 384, 384) float [0, 1]
        # Representative RGB frames (1 per temporal tubelet, 8 total) resized to 224x224 for compact storage
        vj_frames = vjepa_t[0, :, ::2].permute(1, 0, 2, 3)  # (8, 3, 384, 384)
        vj_rgb = (F.interpolate(vj_frames, size=(224, 224), mode="bilinear", align_corners=False)
                  .permute(0, 2, 3, 1).numpy() * 255).clip(0, 255).astype(np.uint8)
        clips_data[vid] = {
            "tag": tag,
            "label": int(r.label),
            "source_group_id": r.source_group_id,
            "dino_rgb": (dino_t[0].permute(0, 2, 3, 1).numpy() * 255).clip(0, 255).astype(np.uint8),  # (4, 224, 224, 3)
            "vjepa_rgb": vj_rgb,  # (8, 224, 224, 3) uint8 (1 frame per tubelet)
            "dino_t": dino_t,
            "vjepa_t": vjepa_t,
        }

    # --- DINOv3 Frozen & LoRA ---
    print("Running DINOv3 Frozen...")
    dino_frozen = DinoBranch(args.dino_weights, pretrained=True, use_lora=False).to(device).eval()
    for vid, _ in selected:
        amap, _, logit = extract_dino_attention(dino_frozen, clips_data[vid]["dino_t"], device)
        clips_data[vid]["dino_frozen_attn"] = amap
        clips_data[vid]["dino_frozen_logit"] = logit
    del dino_frozen
    torch.cuda.empty_cache()

    print("Running DINOv3 LoRA...")
    ckpt_d = torch.load(args.dino_ckpt, map_location="cpu", weights_only=False)
    lora_d = ckpt_d.get("lora") or {}
    dino_lora = DinoBranch(
        args.dino_weights,
        pretrained=True,
        use_lora=True,
        lora_rank=int(lora_d.get("rank", 8)),
        lora_alpha=int(lora_d.get("alpha", 16)),
        lora_dropout=float(lora_d.get("dropout", 0.05)),
        lora_targets=tuple(lora_d["targets"]) if lora_d.get("targets") else None,
    ).to(device).eval()
    state_d = ckpt_d.get("state_dict", ckpt_d)
    miss_d, unexp_d = dino_lora.load_state_dict(state_d, strict=False)
    print(f"  DINO LoRA loaded: missing={len(miss_d)} unexpected={len(unexp_d)}")
    for vid, _ in selected:
        amap, per_head, logit = extract_dino_attention(dino_lora, clips_data[vid]["dino_t"], device)
        clips_data[vid]["dino_lora_attn"] = amap
        clips_data[vid]["dino_lora_heads_f0"] = per_head
        clips_data[vid]["dino_lora_logit"] = logit
    del dino_lora
    torch.cuda.empty_cache()

    # --- V-JEPA 2.1 Frozen & LoRA ---
    print("Running V-JEPA 2.1 Frozen...")
    vjepa_frozen = VJepaBranch(args.vjepa_weights, pretrained=True, use_lora=False).to(device).eval()
    for vid, _ in selected:
        amap, logit = extract_vjepa_attention(vjepa_frozen, clips_data[vid]["vjepa_t"], device)
        clips_data[vid]["vjepa_frozen_attn"] = amap
        clips_data[vid]["vjepa_frozen_logit"] = logit
    del vjepa_frozen
    torch.cuda.empty_cache()

    print("Running V-JEPA 2.1 LoRA...")
    ckpt_v = torch.load(args.vjepa_ckpt, map_location="cpu", weights_only=False)
    lora_v = ckpt_v.get("lora") or {}
    vjepa_lora = VJepaBranch(
        args.vjepa_weights,
        pretrained=True,
        use_lora=True,
        lora_rank=int(lora_v.get("rank", 8)),
        lora_alpha=int(lora_v.get("alpha", 16)),
        lora_dropout=float(lora_v.get("dropout", 0.05)),
        lora_targets=tuple(lora_v["targets"]) if lora_v.get("targets") else None,
    ).to(device).eval()
    state_v = ckpt_v.get("state_dict", ckpt_v)
    miss_v, unexp_v = vjepa_lora.load_state_dict(state_v, strict=False)
    print(f"  V-JEPA LoRA loaded: missing={len(miss_v)} unexpected={len(unexp_v)}")
    for vid, _ in selected:
        amap, logit = extract_vjepa_attention(vjepa_lora, clips_data[vid]["vjepa_t"], device)
        clips_data[vid]["vjepa_lora_attn"] = amap
        clips_data[vid]["vjepa_lora_logit"] = logit
    del vjepa_lora
    torch.cuda.empty_cache()

    # Pack compact NPZ + JSON metadata summary
    npz_payload = {}
    summary = []
    for vid, tag in selected:
        c = clips_data[vid]
        stem = Path(vid).stem
        npz_payload[f"{stem}__dino_rgb"] = c["dino_rgb"]
        npz_payload[f"{stem}__vjepa_rgb"] = c["vjepa_rgb"]
        npz_payload[f"{stem}__dino_frozen_attn"] = c["dino_frozen_attn"].astype(np.float32)
        npz_payload[f"{stem}__dino_lora_attn"] = c["dino_lora_attn"].astype(np.float32)
        npz_payload[f"{stem}__dino_lora_heads_f0"] = c["dino_lora_heads_f0"].astype(np.float32)
        npz_payload[f"{stem}__vjepa_frozen_attn"] = c["vjepa_frozen_attn"].astype(np.float32)
        npz_payload[f"{stem}__vjepa_lora_attn"] = c["vjepa_lora_attn"].astype(np.float32)

        df_mean = c["dino_frozen_attn"].mean(axis=0)
        dl_mean = c["dino_lora_attn"].mean(axis=0)
        vf_mean = c["vjepa_frozen_attn"].mean(axis=0)
        vl_mean = c["vjepa_lora_attn"].mean(axis=0)

        dino_shift_l1 = float(np.abs(dl_mean / (dl_mean.sum() + 1e-12) - df_mean / (df_mean.sum() + 1e-12)).sum())
        vjepa_shift_l1 = float(np.abs(vl_mean / (vl_mean.sum() + 1e-12) - vf_mean / (vf_mean.sum() + 1e-12)).sum())

        # Spatial correlation between DINOv3 (14x14) and V-JEPA (24x24 -> 14x14)
        vl_14 = F.interpolate(torch.from_numpy(vl_mean)[None, None], size=(14, 14), mode="bilinear", align_corners=False)[0, 0].numpy()
        spatial_corr = float(np.corrcoef(dl_mean.ravel(), vl_14.ravel())[0, 1])

        rec = {
            "video_id": vid,
            "stem": stem,
            "tag": tag,
            "label": c["label"],
            "source_group_id": c["source_group_id"],
            "dino_frozen_prob": float(1.0 / (1.0 + np.exp(-c["dino_frozen_logit"]))),
            "dino_lora_prob": float(1.0 / (1.0 + np.exp(-c["dino_lora_logit"]))),
            "vjepa_frozen_prob": float(1.0 / (1.0 + np.exp(-c["vjepa_frozen_logit"]))),
            "vjepa_lora_prob": float(1.0 / (1.0 + np.exp(-c["vjepa_lora_logit"]))),
            "dino_frozen_stats": attention_stats(df_mean),
            "dino_lora_stats": attention_stats(dl_mean),
            "vjepa_frozen_stats": attention_stats(vf_mean),
            "vjepa_lora_stats": attention_stats(vl_mean),
            "dino_shift_l1": dino_shift_l1,
            "vjepa_shift_l1": vjepa_shift_l1,
            "dino_vjepa_spatial_corr": spatial_corr,
            "vjepa_lora_temporal_energy": [float(x) for x in c["vjepa_lora_attn"].sum(axis=(1, 2))],
        }
        summary.append(rec)
        print(
            f"  {vid} [{tag:20s}] y={c['label']} | "
            f"DINO_L={rec['dino_lora_prob']:.3f} (ent {rec['dino_frozen_stats']['norm_entropy']:.3f}->{rec['dino_lora_stats']['norm_entropy']:.3f}, L1={dino_shift_l1:.3f}) | "
            f"VJEPA_L={rec['vjepa_lora_prob']:.3f} (ent {rec['vjepa_frozen_stats']['norm_entropy']:.3f}->{rec['vjepa_lora_stats']['norm_entropy']:.3f}, L1={vjepa_shift_l1:.3f}) | "
            f"spatial_r={spatial_corr:+.3f}"
        )

    np.savez_compressed(out_dir / "attention_maps.npz", **npz_payload)
    (out_dir / "attention_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {out_dir / 'attention_maps.npz'} ({(out_dir / 'attention_maps.npz').stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()

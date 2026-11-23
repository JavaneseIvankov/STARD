"""STAF-Net Live Inference Demo Server (FastAPI + RTX 3060 GPU).

Supports two clean demo inputs:
  1. Stock DFDC test videos (clickable gallery)
  2. Live camera recording from the browser webcam

Runs real-time preprocessing (Haar face crop + 16-frame sampling), DINOv3 ViT-S/16 LoRA,
V-JEPA 2.1 ViT-B/16 LoRA, Standardized Dual-Branch Fusion, and live last-block attention
map overlays on GPU.
"""

from __future__ import annotations

import base64
import io
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from src.contracts import DINO_SIZE, VJEPA_FRAMES, VJEPA_SIZE
from src.data.transforms import DINO_FRAME_INDICES
from src.models.dino_branch import DinoBranch
from src.models.vjepa_branch import VJepaBranch

ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = Path(__file__).resolve().parent
STATIC_DIR = DEMO_DIR / "static"
STOCK_DIR = Path(os.environ.get("STAFNET_STOCK_DIR", str(DEMO_DIR / "stock_videos")))
WEIGHTS_DIR = Path(os.environ.get("STAFNET_WEIGHTS_DIR", str(ROOT / "weights")))
ADAPTERS_DIR = Path(os.environ.get("STAFNET_ADAPTERS_DIR", str(DEMO_DIR / "adapters")))
BUNDLE_PATH = Path(os.environ.get("STAFNET_BUNDLE_PATH", str(DEMO_DIR / "fusion_bundle.pt")))

STOCK_CATALOG = [
    {
        "id": "wkwwpmocfo.mp4",
        "title": "Sample 01 — Strong Dual-Artifact Swap",
        "subtitle": "DFDC Test Clip · Both spatial & temporal cues",
        "ground_truth": "FAKE",
    },
    {
        "id": "llmftzgeec.mp4",
        "title": "Sample 02 — Spatial-Dominant Swap",
        "subtitle": "DFDC Test Clip · Caught primarily by DINOv3",
        "ground_truth": "FAKE",
    },
    {
        "id": "yednyneaag.mp4",
        "title": "Sample 03 — Temporal-Dominant Swap",
        "subtitle": "DFDC Test Clip · Caught primarily by V-JEPA 2.1",
        "ground_truth": "FAKE",
    },
    {
        "id": "mxgdaffjmt.mp4",
        "title": "Sample 04 — High-Fidelity Face Swap",
        "subtitle": "DFDC Test Clip · Subtle blending boundaries",
        "ground_truth": "FAKE",
    },
    {
        "id": "ogghhskfdr.mp4",
        "title": "Sample 05 — Authentic Speaker A",
        "subtitle": "DFDC Test Clip · Unmanipulated real video",
        "ground_truth": "REAL",
    },
    {
        "id": "eobjdumcjj.mp4",
        "title": "Sample 06 — Authentic Speaker B",
        "subtitle": "DFDC Test Clip · Unmanipulated real video",
        "ground_truth": "REAL",
    },
]

app = FastAPI(title="STAF-Net Live Deepfake Detection Demo")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_LOCK = threading.Lock()
_ENGINE: dict[str, object] = {}


def _unwrap(module: torch.nn.Module) -> torch.nn.Module:
    return module.get_base_model() if hasattr(module, "get_base_model") else module


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0))))


def _load_engine() -> dict[str, object]:
    if _ENGINE:
        return _ENGINE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[STAF-Net Demo] Loading models onto {device}...", flush=True)
    t0 = time.perf_counter()

    # Load DINOv3 ViT-S/16 + slim LoRA adapter
    dino_ckpt = torch.load(ADAPTERS_DIR / "dino_lora_adapter.pt", map_location="cpu", weights_only=False)
    lora_d = dino_ckpt.get("lora") or {}
    dino = DinoBranch(
        WEIGHTS_DIR / "dinov3-vits16",
        pretrained=True,
        use_lora=True,
        lora_rank=int(lora_d.get("rank", 8)),
        lora_alpha=int(lora_d.get("alpha", 16)),
        lora_dropout=float(lora_d.get("dropout", 0.05)),
        lora_targets=tuple(lora_d["targets"]) if lora_d.get("targets") else None,
    )
    dino.load_state_dict(dino_ckpt["state_dict"], strict=False)
    dino = dino.to(device).eval()

    # Load V-JEPA 2.1 ViT-B/16 + slim LoRA adapter
    vj_ckpt = torch.load(ADAPTERS_DIR / "vjepa_lora_adapter.pt", map_location="cpu", weights_only=False)
    lora_v = vj_ckpt.get("lora") or {}
    vjepa = VJepaBranch(
        WEIGHTS_DIR / "vjepa2.1-vitb16",
        pretrained=True,
        use_lora=True,
        lora_rank=int(lora_v.get("rank", 8)),
        lora_alpha=int(lora_v.get("alpha", 16)),
        lora_dropout=float(lora_v.get("dropout", 0.05)),
        lora_targets=tuple(lora_v["targets"]) if lora_v.get("targets") else None,
    )
    vjepa.load_state_dict(vj_ckpt["state_dict"], strict=False)
    vjepa = vjepa.to(device).eval()

    bundle = torch.load(BUNDLE_PATH, map_location="cpu", weights_only=False)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    _ENGINE.update({
        "device": device,
        "dino": dino,
        "vjepa": vjepa,
        "bundle": bundle,
        "cascade": cascade,
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
    })
    print(f"[STAF-Net Demo] Ready in {time.perf_counter() - t0:.1f}s on {_ENGINE['gpu_name']}.", flush=True)
    return _ENGINE


def _ensure_mp4(raw_path: Path, work_dir: Path) -> Path:
    """Transcode webcam webm/mov or unusual codecs to standard H.264 mp4 via ffmpeg."""
    cap = cv2.VideoCapture(str(raw_path))
    ok = cap.isOpened()
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if ok else 0
    cap.release()
    if ok and n_frames >= 8 and raw_path.suffix.lower() == ".mp4":
        return raw_path

    out_mp4 = work_dir / "normalized.mp4"
    cmd = [
        "ffmpeg", "-y", "-i", str(raw_path),
        "-an", "-vf", "scale='min(720,iw)':-2",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        str(out_mp4),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not out_mp4.exists():
        raise HTTPException(status_code=400, detail="Could not decode video stream.")
    return out_mp4


def _decode_and_crop_16_frames(video_path: Path, cascade: cv2.CascadeClassifier) -> tuple[np.ndarray, bool]:
    """Decode up to 2.0s window -> 16 frames -> Haar median face crop (384x384 RGB uint8)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Failed to open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if not np.isfinite(fps) or fps <= 1.0 or fps > 120.0:
        fps = 25.0

    # Read up to first ~3 seconds (capped at 120 frames for speed)
    frames_bgr: list[np.ndarray] = []
    max_read = min(int(fps * 3.0), 120)
    while len(frames_bgr) < max_read:
        ret, frame = cap.read()
        if not ret:
            break
        frames_bgr.append(frame)
    cap.release()

    if len(frames_bgr) == 0:
        raise HTTPException(status_code=400, detail="Video contains 0 readable frames.")

    # Pick a 2.0s span (or full clip if shorter) and sample 16 ordered indices
    window_n = min(len(frames_bgr), max(16, int(round(fps * 2.0))))
    idxs = np.linspace(0, window_n - 1, VJEPA_FRAMES).round().astype(int)
    sampled = [frames_bgr[i] for i in idxs]

    # Detect largest face across sampled frames (every 2nd frame for speed)
    boxes = []
    for fr in sampled[::2]:
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(48, 48))
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            boxes.append((x, y, w, h))

    H, W = sampled[0].shape[:2]
    face_detected = len(boxes) > 0
    if face_detected:
        mx, my, mw, mh = np.median(np.array(boxes, dtype=np.float32), axis=0)
        cx, cy = mx + mw / 2.0, my + mh / 2.0
        side = max(mw, mh) * 1.40  # 0.20 margin on each side matching face_tracking.py
    else:
        # Graceful fallback for live webcam demos if face is angled/occluded
        cx, cy = W / 2.0, H / 2.0
        side = min(W, H) * 0.72

    half = side / 2.0
    x0 = max(0, int(round(cx - half)))
    y0 = max(0, int(round(cy - half)))
    x1 = min(W, int(round(cx + half)))
    y1 = min(H, int(round(cy + half)))
    if (x1 - x0) < 32 or (y1 - y0) < 32:
        x0, y0, x1, y1 = 0, 0, W, H

    crops_rgb = []
    for fr in sampled:
        crop = fr[y0:y1, x0:x1]
        crop = cv2.resize(crop, (VJEPA_SIZE, VJEPA_SIZE), interpolation=cv2.INTER_AREA)
        crops_rgb.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    return np.stack(crops_rgb, axis=0), face_detected  # (16, 384, 384, 3) uint8


def _overlay_to_b64(rgb_224: np.ndarray, amap: np.ndarray | None, cmap_name: str = "inferno") -> str:
    if amap is None:
        img = Image.fromarray(rgb_224)
    else:
        t = torch.from_numpy(amap.astype(np.float32))[None, None]
        up = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)[0, 0].numpy()
        lo, hi = np.percentile(up, 2), np.percentile(up, 99.5)
        norm = np.clip((up - lo) / (hi - lo + 1e-8), 0.0, 1.0)
        heat = (matplotlib.colormaps[cmap_name](norm)[:, :, :3] * 255.0).astype(np.float32)
        blend = np.clip(0.45 * rgb_224.astype(np.float32) + 0.55 * heat, 0, 255).astype(np.uint8)
        img = Image.fromarray(blend)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


@torch.no_grad()
def _run_inference(crops_rgb: np.ndarray) -> dict[str, object]:
    eng = _load_engine()
    device: torch.device = eng["device"]
    dino: DinoBranch = eng["dino"]
    vjepa: VJepaBranch = eng["vjepa"]
    bundle: dict = eng["bundle"]

    t0 = time.perf_counter()

    # Build V-JEPA tensor (1, 3, 16, 384, 384) float [0, 1]
    clip_tchw = torch.from_numpy(crops_rgb).permute(0, 3, 1, 2).float() / 255.0
    vj_in = clip_tchw.permute(1, 0, 2, 3).unsqueeze(0).to(device)

    # Build DINO tensor (1, 4, 3, 224, 224) float [0, 1]
    d_sel = clip_tchw[list(DINO_FRAME_INDICES)]
    d_224 = F.interpolate(d_sel, size=(DINO_SIZE, DINO_SIZE), mode="bicubic", align_corners=False).clamp(0.0, 1.0)
    dino_in = d_224.unsqueeze(0).to(device)

    # --- DINOv3 forward + last-block CLS->patch attention ---
    d_base = _unwrap(dino.backbone)
    d_attn_mod = d_base.blocks[-1].attn
    orig_fused = getattr(d_attn_mod, "fused_attn", True)
    d_attn_mod.fused_attn = False
    d_cap: list[torch.Tensor] = []

    def _d_hook(_m, _i, out):
        d_cap.append(out.detach().float().cpu())

    h_d = d_attn_mod.attn_drop.register_forward_hook(_d_hook)
    try:
        with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            z_d = dino.encode(dino_in).z.squeeze(0).float().cpu().numpy()
    finally:
        h_d.remove()
        d_attn_mod.fused_attn = orig_fused

    attn_d = d_cap[0]  # (4, H, 201, 201)
    prefix = attn_d.shape[-1] - 196
    dino_map = attn_d[:, :, 0, prefix:].mean(dim=(0, 1)).reshape(14, 14).numpy()

    # --- V-JEPA 2.1 forward + last-block received spatiotemporal attention ---
    v_base = _unwrap(vjepa.backbone)
    v_attn_mod = v_base.encoder.layer[-1].attention
    orig_v_fwd = v_attn_mod.forward
    v_cap: list[torch.Tensor] = []

    def _v_capture(hidden_states, position_mask=None, T=None, H_patches=None, W_patches=None, output_attentions=False):
        B, N, C = hidden_states.shape
        q = v_attn_mod.query(hidden_states).view(B, N, v_attn_mod.num_attention_heads, v_attn_mod.attention_head_size).transpose(1, 2)
        k = v_attn_mod.key(hidden_states).view(B, N, v_attn_mod.num_attention_heads, v_attn_mod.attention_head_size).transpose(1, 2)
        if hasattr(v_attn_mod, "apply_rotary_embeddings"):
            ids = position_mask.unsqueeze(1).repeat(1, v_attn_mod.num_attention_heads, 1) if position_mask is not None else torch.arange(N, device=hidden_states.device)
            q = v_attn_mod.apply_rotary_embeddings(q, ids, T=T, H_patches=H_patches, W_patches=W_patches)
            k = v_attn_mod.apply_rotary_embeddings(k, ids, T=T, H_patches=H_patches, W_patches=W_patches)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (v_attn_mod.attention_head_size ** -0.5)
        probs = torch.softmax(scores.float(), dim=-1)
        v_cap.append(probs.mean(dim=(1, 2)).squeeze(0).cpu())
        return orig_v_fwd(hidden_states, position_mask=position_mask, T=T, H_patches=H_patches, W_patches=W_patches, output_attentions=output_attentions)

    v_attn_mod.forward = _v_capture
    try:
        with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            z_j = vjepa.encode(vj_in).z.squeeze(0).float().cpu().numpy()
    finally:
        v_attn_mod.forward = orig_v_fwd

    rec_v = v_cap[0].numpy()
    t_grid = rec_v.size // (24 * 24)
    vjepa_map = rec_v.reshape(t_grid, 24, 24).mean(axis=0)

    # --- Standardized Dual-Branch Fusion (0.9030 Test AUROC) ---
    zd_std = (z_d - bundle["dino_mean"]) / np.maximum(bundle["dino_scale"], 1e-6)
    zj_std = (z_j - bundle["vjepa_mean"]) / np.maximum(bundle["vjepa_scale"], 1e-6)
    z_cat = np.concatenate([zd_std, zj_std], axis=0)

    dh = bundle["dino_head"]
    jh = bundle["vjepa_head"]
    fh = bundle["fusion_head"]

    p_dino = _sigmoid(float(np.dot(zd_std, dh["coef"]) + dh["intercept"]))
    p_vjepa = _sigmoid(float(np.dot(zj_std, jh["coef"]) + jh["intercept"]))
    p_fusion = _sigmoid(float(np.dot(z_cat, fh["coef"]) + fh["intercept"]))
    verdict = "FAKE" if p_fusion >= fh["threshold"] else "REAL"

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    rgb_224 = (d_224[0].permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return {
        "verdict": verdict,
        "fusion_prob": round(p_fusion, 4),
        "fusion_threshold": round(float(fh["threshold"]), 3),
        "vjepa_prob": round(p_vjepa, 4),
        "dino_prob": round(p_dino, 4),
        "inference_ms": elapsed_ms,
        "gpu": eng["gpu_name"],
        "face_crop_b64": _overlay_to_b64(rgb_224, None),
        "dino_attn_b64": _overlay_to_b64(rgb_224, dino_map, cmap_name="inferno"),
        "vjepa_attn_b64": _overlay_to_b64(rgb_224, vjepa_map, cmap_name="viridis"),
    }


@app.on_event("startup")
def _startup() -> None:
    _load_engine()


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/api/health")
def health() -> dict[str, object]:
    eng = _load_engine()
    return {"status": "ok", "gpu": eng["gpu_name"], "stock_count": len(list(STOCK_DIR.glob("*.mp4")))}


@app.get("/api/stock")
def list_stock() -> list[dict[str, object]]:
    items = []
    for item in STOCK_CATALOG:
        if (STOCK_DIR / item["id"]).exists():
            items.append({**item, "url": f"/stock/{item['id']}"})
    return items


@app.get("/stock/{filename}")
def get_stock_video(filename: str) -> FileResponse:
    safe_name = Path(filename).name
    path = STOCK_DIR / safe_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Stock video not found.")
    return FileResponse(str(path), media_type="video/mp4")


@app.post("/api/predict")
async def predict(
    stock_id: str | None = Form(default=None),
    video: UploadFile | None = File(default=None),
) -> JSONResponse:
    if not stock_id and video is None:
        raise HTTPException(status_code=400, detail="Provide either stock_id or a video upload.")

    eng = _load_engine()
    with tempfile.TemporaryDirectory(prefix="stafnet_") as tmp:
        tmp_dir = Path(tmp)
        gt = None
        source_label = ""
        if stock_id:
            safe_name = Path(stock_id).name
            vid_path = STOCK_DIR / safe_name
            if not vid_path.exists():
                raise HTTPException(status_code=404, detail=f"Stock video {safe_name} not found.")
            source_label = safe_name
            for item in STOCK_CATALOG:
                if item["id"] == safe_name:
                    gt = item["ground_truth"]
                    source_label = item["title"]
                    break
        else:
            assert video is not None
            suffix = Path(video.filename or "camera.webm").suffix or ".webm"
            raw_path = tmp_dir / f"upload{suffix}"
            raw_path.write_bytes(await video.read())
            vid_path = _ensure_mp4(raw_path, tmp_dir)
            source_label = "Live Camera Capture" if "camera" in (video.filename or "").lower() else (video.filename or "Uploaded Video")

        crops_rgb, face_detected = _decode_and_crop_16_frames(vid_path, eng["cascade"])
        with _LOCK:
            result = _run_inference(crops_rgb)

        result["source"] = source_label
        result["face_detected"] = face_detected
        result["ground_truth"] = gt
        return JSONResponse(result)

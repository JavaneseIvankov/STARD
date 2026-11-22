"""STAF-Net | src/data/face_tracking.py

Owner   : Person 1 + Person 2 (shared). Owner of record: Person 1.
Status  : IMPLEMENTED - OpenCV-only fallback; RetinaFace hook documented.
          NOT yet run on real DFDC videos (needs the dataset on the server).

Responsibility
--------------
For each manifest row, decode the video once and cache one 16-frame facial clip
so Stage I never touches video decoding again:

decode -> pick a ~2 s window -> sample 16 ordered frames -> detect a face ->
track it -> square crop with margin -> resize -> save tensor

The same cached tensor feeds both branches (DINO takes frames 0,5,10,15; V-JEPA
takes all 16), which is what makes A1 vs A3 a controlled comparison.

Output
------
data/clips/<split>/<video_id>.pt
    {
      "frames": FloatTensor (16, 3, H, W) in [0, 1],
      "clip_start_sec": float,
      "preprocessing_version": "clips-v1",
      "face_box": (4,) int tensor of the median tracked box,
      "track_ok": bool,
    }
data/manifests/preprocess_report.json
    decode failures, face-detection failures, per-video status

Detector note
-------------
OpenCV Haar cascade ships with opencv-python and needs no model download, so it
is the default here to keep the pipeline runnable on a fresh server. RetinaFace
(plan 3) is better on profile/small faces; wire it in `_build_detector` and keep
the fallback so a missing model never blocks preprocessing.

Failure policy
--------------
A video that fails to decode or has no detectable face is recorded as failed and
skipped. Silent drops are forbidden: the report must explain every missing clip,
because a biased drop (e.g. only small faces fail) is a real result.

CLI
---
    python -m src.data.face_tracking --config configs/dataset.yaml
    python -m src.data.face_tracking --limit 20 --overwrite
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import torch

from src.contracts import VJEPA_FRAMES, ManifestRow

__all__ = [
    "PREPROCESSING_VERSION",
    "FaceTrackConfig",
    "PreprocessReport",
    "VideoDecodeError",
    "NoFaceError",
    "sample_indices",
    "crop_face_window",
    "process_video",
    "run",
]

PREPROCESSING_VERSION = "clips-v1"


class VideoDecodeError(RuntimeError):
    """Raised when a video cannot be opened or has too few frames."""


class NoFaceError(RuntimeError):
    """Raised when no face is detected in enough sampled frames."""


@dataclass(frozen=True)
class FaceTrackConfig:
    """Preprocessing knobs mirroring plan section 3."""

    num_frames: int = VJEPA_FRAMES
    window_sec: float = 2.0
    margin: float = 0.20
    output_size: int = 384
    min_face_px: int = 48
    clip_start_hint: float | None = None
    detector: str = "haar"  # haar | retinaface
    deterministic: bool = False

    def __post_init__(self) -> None:
        if self.num_frames != VJEPA_FRAMES:
            raise ValueError(
                f"FaceTrackConfig.num_frames: contract requires {VJEPA_FRAMES}, got {self.num_frames}"
            )
        if not 0.0 <= self.margin <= 1.0:
            raise ValueError(f"FaceTrackConfig.margin: must be in [0, 1], got {self.margin}")
        if self.output_size < 224:
            raise ValueError(
                f"FaceTrackConfig.output_size: must be >= 224 (crops are downscaled to "
                f"224 for DINO and upscaled to 384 for V-JEPA), got {self.output_size}"
            )


def sample_indices(total_frames: int, fps: float, cfg: FaceTrackConfig, seed: int = 0) -> list[int]:
    """Pick a contiguous ~2 s window and 16 monotonically increasing frame indices.

    A contiguous window keeps the clip physically plausible; frames are never
    sampled from unrelated parts of the video (plan 3).
    """
    if total_frames <= cfg.num_frames:
        return list(range(total_frames))

    span = int(round(cfg.window_sec * max(fps, 1.0)))
    span = min(max(span, cfg.num_frames), total_frames)

    if cfg.deterministic:
        start = 0
    else:
        latest = max(0, total_frames - span)
        if cfg.clip_start_hint is not None:
            hint_frame = int(cfg.clip_start_hint * max(fps, 1.0))
            start = min(max(0, hint_frame), latest)
        else:
            generator = torch.Generator().manual_seed(seed)
            start = int(torch.randint(0, latest + 1, (1,), generator=generator).item())

    if span == cfg.num_frames:
        return [start + i for i in range(cfg.num_frames)]

    step = span / cfg.num_frames
    indices = [min(start + int(round(i * step)), total_frames - 1) for i in range(cfg.num_frames)]
    return sorted(dict.fromkeys(indices))


def _square_box(
    box: tuple[int, int, int, int], margin: float, shape: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Expand a face box into a square crop with context margin, clipped to the image."""
    x, y, w, h = box
    height, width = shape
    side = int(round(max(w, h) * (1.0 + 2 * margin)))
    cx, cy = x + w / 2.0, y + h / 2.0
    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))
    x0 = max(0, min(x0, max(0, width - side)))
    y0 = max(0, min(y0, max(0, height - side)))
    side = min(side, width - x0, height - y0)
    if side <= 0:
        raise NoFaceError("degenerate face box after margin expansion")
    return x0, y0, side, side


def crop_face_window(frames: list, boxes: list[tuple[int, int, int, int]], cfg: FaceTrackConfig):
    """Crop, resize and stack frames into (16, 3, S, S) float [0, 1].

    `frames` are BGR uint8 arrays, `boxes` are (x, y, w, h). The median box is
    used for every frame so the crop does not jitter across the clip.
    """
    import cv2
    import numpy as np

    if not frames:
        raise VideoDecodeError("no frames decoded")
    arr = np.asarray(boxes, dtype=np.float32)
    median = tuple(int(v) for v in np.median(arr, axis=0))
    height, width = frames[0].shape[:2]
    x0, y0, side, _ = _square_box(median, cfg.margin, (height, width))

    out = []
    for frame in frames:
        crop = frame[y0 : y0 + side, x0 : x0 + side]
        if crop.size == 0:
            raise VideoDecodeError("empty crop")
        resized = cv2.resize(crop, (cfg.output_size, cfg.output_size), interpolation=cv2.INTER_AREA)
        out.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

    tensor = torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2).float() / 255.0
    return tensor.clamp(0.0, 1.0), median


def _build_detector(cfg: FaceTrackConfig):
    """Return a callable frame -> list of (x, y, w, h).

    `retinaface` is wired but not the default: it needs weights that are not
    vendored here. Swap the body in when the weights are present on the server.
    """
    import cv2

    if cfg.detector == "retinaface":
        raise NotImplementedError(
            "face_tracking: retinaface detector not wired yet. Install the weights and "
            "implement _build_detector(..., detector='retinaface'), or keep 'haar'."
        )

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError(f"face_tracking: cannot load Haar cascade at {cascade_path}")

    def detect(frame) -> list[tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(cfg.min_face_px, cfg.min_face_px)
        )
        return [tuple(int(v) for v in box) for box in faces]

    return detect


def process_video(row: ManifestRow, cfg: FaceTrackConfig, *, seed: int = 0) -> dict[str, object]:
    """Decode one video into a cached 16-frame facial clip payload."""
    import cv2

    path = Path(row.filepath)
    if not path.exists():
        raise VideoDecodeError(f"video not found: {path}")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoDecodeError(f"cannot open video: {path}")
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if total < cfg.num_frames:
            raise VideoDecodeError(
                f"too few frames ({total}) for a {cfg.num_frames}-frame clip"
            )
        indices = sample_indices(total, fps or 25.0, cfg, seed=seed)

        wanted = set(indices)
        collected: dict[int, object] = {}
        index = 0
        while len(collected) < len(wanted):
            grabbed, frame = capture.read()
            if not grabbed:
                break
            if index in wanted:
                collected[index] = frame
            index += 1
    finally:
        capture.release()

    missing = [i for i in indices if i not in collected]
    if missing:
        raise VideoDecodeError(f"failed to decode frames {missing[:5]}")

    frames = [collected[i] for i in indices]
    detect = _build_detector(cfg)

    boxes: list[tuple[int, int, int, int]] = []
    kept_frames: list = []
    for frame in frames:
        found = detect(frame)
        if not found:
            continue
        boxes.append(max(found, key=lambda b: b[2] * b[3]))
        kept_frames.append(frame)

    if len(boxes) < max(3, cfg.num_frames // 2):
        raise NoFaceError(
            f"face detected in only {len(boxes)}/{cfg.num_frames} frames of {path.name}"
        )

    clip, median_box = crop_face_window(kept_frames, boxes, cfg)
    start_sec = indices[0] / (fps or 25.0)

    return {
        "frames": clip,
        "clip_start_sec": float(start_sec),
        "preprocessing_version": PREPROCESSING_VERSION,
        "face_box": torch.tensor(median_box, dtype=torch.int32),
        "track_ok": len(boxes) == cfg.num_frames,
    }


@dataclass
class PreprocessReport:
    """Explicit accounting of what was written and what was skipped, and why."""

    written: int = 0
    skipped_existing: int = 0
    failed: dict[str, str] = field(default_factory=dict)

    def record_failure(self, video_id: str, reason: str) -> None:
        self.failed[video_id] = reason

    def summary(self) -> dict[str, object]:
        reasons = Counter(reason.split(":")[0].strip() for reason in self.failed.values())
        total = self.written + self.skipped_existing + len(self.failed)
        rate = 0.0 if total == 0 else len(self.failed) / total
        return {
            "preprocessing_version": PREPROCESSING_VERSION,
            "written": self.written,
            "skipped_existing": self.skipped_existing,
            "failed": len(self.failed),
            "failure_rate": round(rate, 4),
            "failure_reasons": dict(reasons),
            "failures": self.failed,
        }


def run(
    rows: list[ManifestRow],
    out_root: str | Path,
    cfg: FaceTrackConfig,
    *,
    overwrite: bool = False,
    limit: int | None = None,
) -> PreprocessReport:
    """Process manifest rows into data/clips/<split>/<video_id>.pt."""
    out_root = Path(out_root)
    report = PreprocessReport()

    for position, row in enumerate(rows):
        if limit is not None and position >= limit:
            break
        target = out_root / row.split / f"{row.video_id}.pt"
        if target.exists() and not overwrite:
            report.skipped_existing += 1
            continue
        try:
            payload = process_video(row, cfg)
        except (VideoDecodeError, NoFaceError) as exc:
            report.record_failure(row.video_id, f"{type(exc).__name__}: {exc}")
            continue
        except Exception as exc:  # noqa: BLE001 - report, never abort the batch
            report.record_failure(row.video_id, f"Unexpected: {exc}")
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, target)
        report.written += 1

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cache 16-frame facial clips from DFDC videos.")
    parser.add_argument("--manifest", default="data/manifests/manifest.csv")
    parser.add_argument("--out", default="data/clips")
    parser.add_argument("--report", default="data/manifests/preprocess_report.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--detector", default="haar", choices=("haar", "retinaface"))
    parser.add_argument("--output-size", type=int, default=384)
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"[wait] manifest not found: {manifest_path}")
        print("       run: python -m src.data.build_manifest --metadata <metadata.json>")
        return 2

    try:
        import cv2  # noqa: F401
    except ModuleNotFoundError:
        print("[env ] opencv is not installed: pip install opencv-python")
        return 3

    from src.data.video_dataset import read_manifest

    rows = read_manifest(manifest_path)
    cfg = FaceTrackConfig(detector=args.detector, output_size=args.output_size)
    report = run(rows, args.out, cfg, overwrite=args.overwrite, limit=args.limit)

    summary = report.summary()
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"clips written={summary['written']} skipped={summary['skipped_existing']} "
        f"failed={summary['failed']} rate={summary['failure_rate']}"
    )
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

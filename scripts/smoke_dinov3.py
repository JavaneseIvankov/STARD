"""STARD | scripts/smoke_dinov3.py

Owner   : Orang A (Spatial Model Lead)
Status  : RUNNABLE - contract checks pass today; extractor checks wait for
          DINOv3SpatialExtractor in src/models/dinov3.py.

Purpose
-------
Smallest possible evidence loop for the spatial branch. Runs without any
dataset, split file, or training loop, so it works while Orang 1 and Orang C
are still building their modules.

What it checks
--------------
1. contracts      : frozen constants are internally consistent, dataclass
                    validators accept correct shapes and reject wrong ones.
2. extractor      : DINOv3SpatialExtractor loads, is frozen, and maps
                    (B, 16, 3, 224, 224) -> (B, 16, 196, 384) float32.
3. behaviour      : deterministic in eval mode, finite outputs, non-constant
                    features, per-batch latency, peak VRAM.

Usage
-----
    python scripts/smoke_dinov3.py --contracts-only
    python scripts/smoke_dinov3.py --batch-size 2
    python scripts/smoke_dinov3.py --batch-size 8 --device cuda
    python scripts/smoke_dinov3.py --weights dinov3-vits16

Exit codes
----------
    0  all requested checks passed
    1  a check failed
    2  extractor not implemented yet (expected while dinov3.py is a placeholder)
    3  environment problem (torch missing, CUDA requested but unavailable)
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_NOT_IMPLEMENTED = 2
EXIT_ENV = 3


def ok(msg: str) -> None:
    print(f"[ ok ] {msg}")


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def skip(msg: str) -> None:
    print(f"[skip] {msg}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test the STARD spatial branch.")
    parser.add_argument("--batch-size", type=int, default=2, help="dummy batch size (default: 2)")
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="device to run on (default: auto -> cuda when available)",
    )
    parser.add_argument("--weights", default=None, help="backbone name/path passed to the extractor")
    parser.add_argument("--iters", type=int, default=3, help="timed forward passes (default: 3)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--contracts-only",
        action="store_true",
        help="only run contract checks, skip the extractor",
    )
    return parser.parse_args(argv)


def check_contracts(torch) -> int:
    """Validate the frozen contract module against synthetic tensors."""
    from src.contracts import (
        D_SPATIAL,
        D_TEMPORAL,
        GRID_SIZE,
        IMAGE_SIZE,
        NUM_CLASSES,
        NUM_FRAMES,
        NUM_PATCHES,
        PATCH_SIZE,
        BackboneFeatures,
        BatchInput,
        ModelOutputs,
        contract_summary,
    )

    failures = 0

    geometry = {
        "PATCH_SIZE * GRID_SIZE == IMAGE_SIZE": PATCH_SIZE * GRID_SIZE == IMAGE_SIZE,
        "GRID_SIZE**2 == NUM_PATCHES": GRID_SIZE**2 == NUM_PATCHES,
        "NUM_PATCHES == 196": NUM_PATCHES == 196,
        "D_SPATIAL == 384": D_SPATIAL == 384,
        "D_TEMPORAL == 768": D_TEMPORAL == 768,
        "NUM_CLASSES == 3": NUM_CLASSES == 3,
    }
    for name, passed in geometry.items():
        if passed:
            ok(f"contract {name}")
        else:
            fail(f"contract {name}")
            failures += 1

    b, t = 2, NUM_FRAMES
    batch = BatchInput(
        frames=torch.rand(b, t, 3, IMAGE_SIZE, IMAGE_SIZE),
        label_cls=torch.randint(0, NUM_CLASSES, (b,)),
        label_spatial=torch.randint(0, 2, (b, t, IMAGE_SIZE, IMAGE_SIZE)).float(),
        label_temporal=torch.randint(0, 2, (b, t)).float(),
    )
    try:
        batch.validate()
        ok("BatchInput.validate() accepts a well-formed batch")
    except Exception as exc:  # noqa: BLE001 - smoke test reports, never raises
        fail(f"BatchInput.validate() rejected a valid batch: {exc}")
        failures += 1

    bad_cases = [
        (
            "frames wrong frame count",
            lambda: BatchInput(
                frames=torch.rand(b, t - 1, 3, IMAGE_SIZE, IMAGE_SIZE),
                label_cls=torch.zeros(b, dtype=torch.long),
            ).validate(),
        ),
        (
            "frames outside [0, 1]",
            lambda: BatchInput(
                frames=torch.rand(b, t, 3, IMAGE_SIZE, IMAGE_SIZE) * 255.0,
                label_cls=torch.zeros(b, dtype=torch.long),
            ).validate(),
        ),
        (
            "float label_cls",
            lambda: BatchInput(
                frames=torch.rand(b, t, 3, IMAGE_SIZE, IMAGE_SIZE),
                label_cls=torch.zeros(b, dtype=torch.float32),
            ).validate(),
        ),
        (
            "label out of range",
            lambda: BatchInput(
                frames=torch.rand(b, t, 3, IMAGE_SIZE, IMAGE_SIZE),
                label_cls=torch.full((b,), NUM_CLASSES, dtype=torch.long),
            ).validate(),
        ),
        (
            "spatial feature wrong token count",
            lambda: BackboneFeatures(
                spatial=torch.rand(b, t, NUM_PATCHES + 60, D_SPATIAL)
            ).validate(),
        ),
    ]
    for name, case in bad_cases:
        try:
            case()
        except (ValueError, TypeError):
            ok(f"contract rejects {name}")
        else:
            fail(f"contract accepted invalid input: {name}")
            failures += 1

    try:
        BackboneFeatures(
            spatial=torch.rand(b, t, NUM_PATCHES, D_SPATIAL),
            temporal=torch.rand(b, t, NUM_PATCHES, D_TEMPORAL),
        ).validate()
        ModelOutputs(
            logits_cls=torch.rand(b, NUM_CLASSES),
            mask_spatial=torch.rand(b, t, IMAGE_SIZE, IMAGE_SIZE),
            mask_temporal=torch.rand(b, t),
        ).validate()
        ok("BackboneFeatures / ModelOutputs validate() accept contract shapes")
    except Exception as exc:  # noqa: BLE001
        fail(f"contract shapes rejected valid tensors: {exc}")
        failures += 1

    print("\ncontract summary:")
    for key, value in contract_summary().items():
        print(f"  {key:22s} = {value}")
    return failures


def resolve_device(torch, requested: str):
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check_extractor(torch, args) -> int:
    from src.contracts import D_SPATIAL, IMAGE_SIZE, NUM_FRAMES, NUM_PATCHES, BackboneFeatures

    try:
        from src.models.dinov3 import DINOv3SpatialExtractor
    except ImportError as exc:
        print(f"\n[wait] cannot import DINOv3SpatialExtractor: {exc}")
        print("       implement src/models/dinov3.py first, then rerun this script.")
        return EXIT_NOT_IMPLEMENTED

    try:
        device = resolve_device(torch, args.device)
    except RuntimeError as exc:
        print(f"\n[env ] {exc}")
        return EXIT_ENV

    failures = 0
    print(f"\ndevice: {device}")

    try:
        if args.weights:
            model = DINOv3SpatialExtractor(pretrained_name=args.weights)
        else:
            model = DINOv3SpatialExtractor()
    except NotImplementedError:
        print("[wait] DINOv3SpatialExtractor.__init__ is still a placeholder.")
        return EXIT_NOT_IMPLEMENTED
    except Exception:  # noqa: BLE001
        print("[FAIL] DINOv3SpatialExtractor() raised:")
        traceback.print_exc()
        return EXIT_FAIL

    model = model.to(device)
    model.eval()
    ok("extractor constructed")

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if trainable:
        fail(f"backbone should be frozen, {len(trainable)} trainable params (e.g. {trainable[0]})")
        failures += 1
    else:
        ok("all backbone parameters frozen")

    if model.training:
        fail("model.training is True after eval() - check for a re-train() call in forward")
        failures += 1
    else:
        ok("model in eval mode")

    frames = torch.rand(args.batch_size, NUM_FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    try:
        with torch.no_grad():
            out = model(frames)
    except NotImplementedError:
        print("[wait] DINOv3SpatialExtractor.forward is still a placeholder.")
        return EXIT_NOT_IMPLEMENTED
    except Exception:  # noqa: BLE001
        print("[FAIL] forward() raised:")
        traceback.print_exc()
        return EXIT_FAIL

    if isinstance(out, dict):
        out = out.get("spatial")

    expected = (args.batch_size, NUM_FRAMES, NUM_PATCHES, D_SPATIAL)
    try:
        BackboneFeatures(spatial=out).validate()
        ok(f"output shape {tuple(out.shape)} matches contract {expected}")
    except Exception as exc:  # noqa: BLE001
        fail(f"output shape contract violation: {exc}")
        failures += 1

    if out.dtype != torch.float32:
        fail(f"output dtype is {out.dtype}, contract requires torch.float32")
        failures += 1
    else:
        ok("output dtype float32")

    if not torch.isfinite(out).all():
        fail("output contains NaN/Inf")
        failures += 1
    else:
        ok("output finite")

    if float(out.std()) == 0.0:
        fail("output is constant - backbone probably not loaded or not normalized")
        failures += 1
    else:
        ok(f"output non-constant (std={float(out.std()):.4f})")

    with torch.no_grad():
        again = model(frames)
    if torch.allclose(out, again, atol=1e-5):
        ok("eval mode is deterministic")
    else:
        fail("two eval forwards differ - check dropout / BN train mode / random aug inside forward")
        failures += 1

    timings: list[float] = []
    for _ in range(max(1, args.iters)):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.no_grad():
            model(frames)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - start)
    mean_ms = 1000.0 * sum(timings) / len(timings)
    print(f"\nlatency: {mean_ms:.1f} ms/batch (B={args.batch_size}, {len(timings)} iters)")

    if device.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3
        print(f"peak VRAM: {peak_gb:.2f} GB")
        print(f"per-video features: {NUM_FRAMES * NUM_PATCHES * D_SPATIAL * 4 / 1024**2:.1f} MiB fp32")
    else:
        print("peak VRAM: n/a (cpu)")

    return EXIT_OK if failures == 0 else EXIT_FAIL


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        import torch
    except ModuleNotFoundError:
        print("[env ] torch is not installed in this interpreter.")
        print("       pip install -r requirements.txt (use Python 3.10-3.12).")
        return EXIT_ENV

    torch.manual_seed(args.seed)

    print("=== contracts ===")
    failures = check_contracts(torch)
    if failures:
        print(f"\n{failures} contract check(s) failed.")
        return EXIT_FAIL

    if args.contracts_only:
        print("\nall contract checks passed (extractor skipped).")
        return EXIT_OK

    print("\n=== extractor ===")
    return check_extractor(torch, args)


if __name__ == "__main__":
    raise SystemExit(main())

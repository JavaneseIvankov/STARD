"""STAF-Net | scripts/smoke_branches.py

Owner   : Person 1 + Person 2 (shared dev tooling)
Status  : RUNNABLE - contract checks pass today; branch checks wait for
          DinoBranch / VJepaBranch in src/models/.

Purpose
-------
Dataset-free evidence loop. Runs before any DFDC download, manifest, or GPU
session, so each person can verify their module against the frozen contract
while the others are still building theirs. This is the 00:00-00:15 sanity
slot of the A100 schedule.

Usage
-----
    python scripts/smoke_branches.py --contracts-only
    python scripts/smoke_branches.py --branch dino  --batch-size 2
    python scripts/smoke_branches.py --branch vjepa --batch-size 1 --device cuda
    python scripts/smoke_branches.py --branch both

Exit codes
----------
    0  all requested checks passed
    1  a check failed
    2  branch not implemented yet (expected while src/models/*_branch.py are placeholders)
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test STAF-Net branches.")
    parser.add_argument("--branch", default="dino", choices=("dino", "vjepa", "both"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--weights", default=None, help="checkpoint id/path for the branch")
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--contracts-only", action="store_true")
    return parser.parse_args(argv)


def check_contracts(torch) -> int:
    from src.contracts import (
        DINO_DIM,
        DINO_FRAMES,
        DINO_SIZE,
        D_PROJ,
        VARIANTS,
        VJEPA_DIM,
        VJEPA_FRAMES,
        VJEPA_SIZE,
        BranchFeatures,
        ClipBatch,
        FeatureRecord,
        FusionInput,
        ManifestRow,
        STAFNetOutput,
        contract_summary,
        variant_branch,
        variant_dim,
    )

    failures = 0

    invariants = {
        "dino geometry 4x224x384": (DINO_FRAMES, DINO_SIZE, DINO_DIM) == (4, 224, 384),
        "vjepa geometry 16x384x768": (VJEPA_FRAMES, VJEPA_SIZE, VJEPA_DIM) == (16, 384, 768),
        "projection dim 256": D_PROJ == 256,
        "four cache variants": VARIANTS == ("dino_frozen", "dino_lora", "vjepa_frozen", "vjepa_lora"),
        "variant_branch('vjepa_lora') == 'vjepa'": variant_branch("vjepa_lora") == "vjepa",
        "variant_dim('dino_frozen') == 384": variant_dim("dino_frozen") == 384,
        "variant_dim('vjepa_lora') == 768": variant_dim("vjepa_lora") == 768,
    }
    for name, passed in invariants.items():
        if passed:
            ok(f"contract {name}")
        else:
            fail(f"contract {name}")
            failures += 1

    b = 2
    batch = ClipBatch(
        label=torch.randint(0, 2, (b,)).float(),
        dino_frames=torch.rand(b, DINO_FRAMES, 3, DINO_SIZE, DINO_SIZE),
        vjepa_frames=torch.rand(b, 3, VJEPA_FRAMES, VJEPA_SIZE, VJEPA_SIZE),
        video_id=tuple(f"vid{i}" for i in range(b)),
    )
    try:
        batch.validate()
        ok("ClipBatch.validate() accepts a well-formed batch")
    except Exception as exc:  # noqa: BLE001
        fail(f"ClipBatch.validate() rejected a valid batch: {exc}")
        failures += 1

    negative_cases = {
        "dino frames with T=16": lambda: ClipBatch(
            label=torch.zeros(b),
            dino_frames=torch.rand(b, 16, 3, DINO_SIZE, DINO_SIZE),
        ).validate(),
        "vjepa frames in (B,T,C,H,W) order": lambda: ClipBatch(
            label=torch.zeros(b),
            vjepa_frames=torch.rand(b, VJEPA_FRAMES, 3, VJEPA_SIZE, VJEPA_SIZE),
        ).validate(),
        "vjepa at 224 instead of 384": lambda: ClipBatch(
            label=torch.zeros(b),
            vjepa_frames=torch.rand(b, 3, VJEPA_FRAMES, 224, 224),
        ).validate(),
        "frames outside [0,1]": lambda: ClipBatch(
            label=torch.zeros(b),
            dino_frames=torch.rand(b, DINO_FRAMES, 3, DINO_SIZE, DINO_SIZE) * 255.0,
        ).validate(),
        "label 2 is not binary": lambda: ClipBatch(
            label=torch.full((b,), 2.0),
            dino_frames=torch.rand(b, DINO_FRAMES, 3, DINO_SIZE, DINO_SIZE),
        ).validate(),
        "no branch set": lambda: ClipBatch(label=torch.zeros(b)).validate(),
        "branch feature wrong width": lambda: BranchFeatures(
            z=torch.rand(b, 768), branch="dino"
        ).validate(),
        "fusion input dim mismatch": lambda: FusionInput(
            label=torch.zeros(b), dino=torch.rand(b, DINO_DIM), vjepa=torch.rand(b + 1, VJEPA_DIM)
        ).validate(),
        "gate outside [0,1]": lambda: STAFNetOutput(
            logit=torch.rand(b, 1), gate=torch.rand(b, D_PROJ) * 4.0
        ).validate(),
        "feature record missing variant": lambda: FeatureRecord(
            video_id="v", label=0, split="train", source_group_id="g",
            features={"dino_frozen": torch.rand(DINO_DIM)},
        ).validate(),
        "manifest split typo": lambda: ManifestRow(
            video_id="v", filepath="p", label=0, split="valid", source_group_id="g"
        ).validate(),
        "manifest label 3": lambda: ManifestRow(
            video_id="v", filepath="p", label=3, split="train", source_group_id="g"
        ).validate(),
    }
    for name, case in negative_cases.items():
        try:
            case()
        except (ValueError, TypeError):
            ok(f"contract rejects {name}")
        else:
            fail(f"contract accepted invalid input: {name}")
            failures += 1

    try:
        BranchFeatures(z=torch.rand(b, DINO_DIM), branch="dino").validate()
        BranchFeatures(z=torch.rand(b, VJEPA_DIM), branch="vjepa").validate()
        FusionInput(
            label=torch.zeros(b), dino=torch.rand(b, DINO_DIM), vjepa=torch.rand(b, VJEPA_DIM)
        ).validate()
        STAFNetOutput(logit=torch.rand(b, 1), gate=torch.rand(b, D_PROJ)).validate()
        FeatureRecord(
            video_id="v", label=1, split="test", source_group_id="g",
            features={
                "dino_frozen": torch.rand(DINO_DIM),
                "dino_lora": torch.rand(DINO_DIM),
                "vjepa_frozen": torch.rand(VJEPA_DIM),
                "vjepa_lora": torch.rand(VJEPA_DIM),
            },
        ).validate()
        ok("all positive cases validate")
    except Exception as exc:  # noqa: BLE001
        fail(f"valid contract tensors were rejected: {exc}")
        failures += 1

    print("\ncontract summary:")
    for key, value in contract_summary().items():
        print(f"  {key:16s} = {value}")
    return failures


def resolve_device(torch, requested: str):
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check_branch(torch, branch: str, args, device) -> int:
    from src.contracts import DINO_DIM, DINO_FRAMES, DINO_SIZE, VJEPA_DIM, VJEPA_FRAMES, VJEPA_SIZE

    if branch == "dino":
        module, class_name = "src.models.dino_branch", "DinoBranch"
        frames = torch.rand(args.batch_size, DINO_FRAMES, 3, DINO_SIZE, DINO_SIZE, device=device)
        expected_dim = DINO_DIM
    else:
        module, class_name = "src.models.vjepa_branch", "VJepaBranch"
        frames = torch.rand(args.batch_size, 3, VJEPA_FRAMES, VJEPA_SIZE, VJEPA_SIZE, device=device)
        expected_dim = VJEPA_DIM

    print(f"\n--- {branch} ---")
    try:
        mod = __import__(module, fromlist=[class_name])
        cls = getattr(mod, class_name)
    except (ImportError, AttributeError) as exc:
        print(f"[wait] cannot import {class_name}: {exc}")
        print(f"       implement {module}.py first, then rerun this script.")
        return EXIT_NOT_IMPLEMENTED

    failures = 0
    try:
        model = cls(pretrained_name=args.weights) if args.weights else cls()
    except NotImplementedError:
        print(f"[wait] {class_name}.__init__ is still a placeholder.")
        return EXIT_NOT_IMPLEMENTED
    except Exception:  # noqa: BLE001
        print(f"[FAIL] {class_name}() raised:")
        traceback.print_exc()
        return EXIT_FAIL

    model = model.to(device)
    model.eval()
    ok(f"{class_name} constructed")

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if trainable:
        fail(f"base weights should be frozen, {len(trainable)} trainable (e.g. {trainable[0]})")
        failures += 1
    else:
        ok("all base parameters frozen")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    try:
        with torch.no_grad():
            out = model(frames)
    except NotImplementedError:
        print(f"[wait] {class_name}.forward is still a placeholder.")
        return EXIT_NOT_IMPLEMENTED
    except Exception:  # noqa: BLE001
        print("[FAIL] forward() raised:")
        traceback.print_exc()
        return EXIT_FAIL

    z = out.z if hasattr(out, "z") else out
    try:
        from src.contracts import BranchFeatures

        BranchFeatures(z=z, branch=branch).validate()
        ok(f"output shape {tuple(z.shape)} matches contract (B, {expected_dim})")
    except Exception as exc:  # noqa: BLE001
        fail(f"output contract violation: {exc}")
        failures += 1

    if z.dtype != torch.float32:
        fail(f"output dtype {z.dtype}, contract requires float32")
        failures += 1
    else:
        ok("output dtype float32")

    if not torch.isfinite(z).all():
        fail("output contains NaN/Inf")
        failures += 1
    else:
        ok("output finite")

    if float(z.std()) == 0.0:
        fail("output constant - weights not loaded or input not normalized")
        failures += 1
    else:
        ok(f"output non-constant (std={float(z.std()):.4f})")

    with torch.no_grad():
        again = model(frames)
    z2 = again.z if hasattr(again, "z") else again
    if torch.allclose(z, z2, atol=1e-5):
        ok("eval mode deterministic")
    else:
        fail("two eval forwards differ - check dropout / BN / random aug inside forward")
        failures += 1

    timings = []
    for _ in range(max(1, args.iters)):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.no_grad():
            model(frames)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - start)
    print(f"latency: {1000 * sum(timings) / len(timings):.1f} ms/batch (B={args.batch_size})")
    if device.type == "cuda":
        print(f"peak VRAM: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB")
    print(f"embedding size: {expected_dim * 4 / 1024:.1f} KiB fp32 per video")

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
        print("\nall contract checks passed (branches skipped).")
        return EXIT_OK

    try:
        device = resolve_device(torch, args.device)
    except RuntimeError as exc:
        print(f"\n[env ] {exc}")
        return EXIT_ENV
    print(f"\ndevice: {device}")

    branches = ["dino", "vjepa"] if args.branch == "both" else [args.branch]
    statuses = [check_branch(torch, branch, args, device) for branch in branches]
    if any(s == EXIT_FAIL for s in statuses):
        return EXIT_FAIL
    if all(s == EXIT_NOT_IMPLEMENTED for s in statuses):
        return EXIT_NOT_IMPLEMENTED
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())

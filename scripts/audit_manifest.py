"""STAF-Net | scripts/audit_manifest.py

Owner   : Person 1
Status  : IMPLEMENTED - run before spending GPU time.

Purpose
-------
Verify a built manifest before it is used for training. build_manifest.py already
refuses to write a split-leaking manifest, but this audit re-checks the written
CSV independently and reports the numbers that belong in the report:

  hard checks (exit 1 on failure)
    1. video_id is unique
    2. label is 0 or 1
    3. split is train / val / test
    4. every filepath exists on disk (unless --no-path-check)
    5. no source group spans two splits
    6. when a fake's original_id is also in the manifest, both sit in the same
       split (this is what makes the split leakage-safe for DFDC pairs)

  reported numbers (never fatal)
    7. per-split real/fake balance
    8. how many fakes actually have their original inside the subset
    9. how many videos were dropped compared to the requested targets
   10. per-split source_part coverage

Usage
-----
    python scripts/audit_manifest.py
    python scripts/audit_manifest.py --manifest data/manifests/manifest.csv
    python scripts/audit_manifest.py --no-path-check
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.contracts import SPLITS  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a STAF-Net manifest before training.")
    parser.add_argument("--manifest", default="data/manifests/manifest.csv")
    parser.add_argument(
        "--metadata",
        default=None,
        help="optional metadata.json/csv for source_part coverage reporting",
    )
    parser.add_argument("--no-path-check", action="store_true")
    parser.add_argument("--expected-total", type=int, default=None, help="plan target, e.g. 1500")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    path = Path(args.manifest)
    if not path.exists():
        print(f"[FAIL] manifest not found: {path}")
        return 1

    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    if not rows:
        print("[FAIL] manifest is empty")
        return 1

    failures: list[str] = []

    # 1. unique ids
    ids = [r["video_id"] for r in rows]
    duplicates = [vid for vid, count in Counter(ids).items() if count > 1]
    if duplicates:
        failures.append(f"duplicate video_id x{len(duplicates)} (e.g. {duplicates[:3]})")

    # 2/3. label + split domains
    bad_labels = [r["video_id"] for r in rows if r["label"] not in {"0", "1"}]
    if bad_labels:
        failures.append(f"invalid label x{len(bad_labels)} (e.g. {bad_labels[:3]})")
    bad_splits = [r["video_id"] for r in rows if r["split"] not in SPLITS]
    if bad_splits:
        failures.append(f"invalid split x{len(bad_splits)} (e.g. {bad_splits[:3]})")

    # 4. paths
    if not args.no_path_check:
        missing = [r["filepath"] for r in rows if not Path(r["filepath"]).exists()]
        if missing:
            failures.append(f"missing files x{len(missing)} (e.g. {missing[:3]})")

    # 5. group atomicity
    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group_splits[row["source_group_id"]].add(row["split"])
    spanning = {g: s for g, s in group_splits.items() if len(s) > 1}
    if spanning:
        failures.append(
            f"source groups spanning splits x{len(spanning)} "
            f"(e.g. {list(spanning.items())[:2]})"
        )

    # 6. fake/original pair atomicity, only when the original is inside the subset
    split_of = {row["video_id"]: row["split"] for row in rows}
    split_pairs = 0
    broken_pairs = []
    for row in rows:
        original = row.get("original_id") or ""
        if not original or original not in split_of:
            continue
        split_pairs += 1
        if split_of[original] != row["split"]:
            broken_pairs.append((row["video_id"], original))
    if broken_pairs:
        failures.append(
            f"fake/original pairs split across partitions x{len(broken_pairs)} "
            f"(e.g. {broken_pairs[:2]})"
        )

    # ---- report ----
    print(f"manifest: {path}  rows={len(rows)}")
    print("\nper-split counts:")
    for split in SPLITS:
        subset = [r for r in rows if r["split"] == split]
        real = sum(1 for r in subset if r["label"] == "0")
        fake = sum(1 for r in subset if r["label"] == "1")
        groups = len({r["source_group_id"] for r in subset})
        ratio = "n/a" if fake == 0 else f"{real / fake:.2f}"
        print(
            f"  {split:5s} real={real:5d} fake={fake:5d} total={len(subset):5d} "
            f"groups={groups:5d} real/fake={ratio}"
        )

    total_real = sum(1 for r in rows if r["label"] == "0")
    total_fake = sum(1 for r in rows if r["label"] == "1")
    print(f"  total real={total_real} fake={total_fake} rows={len(rows)}")

    fakes_with_original = sum(1 for r in rows if r["label"] == "1" and (r.get("original_id") or ""))
    fakes_linked = sum(
        1
        for r in rows
        if r["label"] == "1"
        and (r.get("original_id") or "") in split_of
    )
    print("\nleakage control:")
    print(f"  fakes with an original_id recorded : {fakes_with_original}/{total_fake}")
    print(f"  fakes whose original is in the pool: {fakes_linked}/{total_fake}")
    print(f"  fake/original pairs verified        : {split_pairs}")
    print(
        "  note: fakes whose original is outside the subset cannot leak through "
        "that link; they are unpaired by construction, not by mistake"
    )

    if args.expected_total:
        dropped = args.expected_total - len(rows)
        print("\nplan target:")
        print(f"  expected={args.expected_total} actual={len(rows)} difference={dropped:+d}")
        if dropped:
            print(
                "  note: whole source groups are assigned atomically, and quota "
                "rounding can leave a few videos unassigned; report the actual counts"
            )

    if args.metadata:
        sys.path.insert(0, str(REPO_ROOT))
        from src.data.build_manifest import load_metadata_any

        meta = load_metadata_any(args.metadata)
        coverage: dict[str, Counter] = defaultdict(Counter)
        for row in rows:
            part = str(meta.get(row["video_id"], {}).get("source_part") or "")
            if part:
                coverage[row["split"]][part] += 1
        if coverage:
            print("\nsource archive coverage:")
            for split in SPLITS:
                if coverage[split]:
                    print(f"  {split:5s} parts={len(coverage[split])}")

    if failures:
        print("\nFAILED checks:")
        for failure in failures:
            print(f"  [FAIL] {failure}")
        return 1

    print("\nall hard checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

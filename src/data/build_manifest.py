"""STAF-Net | src/data/build_manifest.py

Owner   : Person 1 + Person 2 (shared). Owner of record: Person 1.
Status  : IMPLEMENTED - pure stdlib; verified on synthetic DFDC metadata.

Responsibility
--------------
Turn DFDC metadata into a leakage-controlled manifest:

1. read the metadata JSON (label / split / original per video),
2. group every fake with its `original` real video transitively into one
   source group,
3. assign whole source groups to train/val/test with a fixed seed,
4. balance real and fake by video count,
5. write data/manifests/manifest.csv and split_summary.json.

DFDC metadata shape (fields we use)
-----------------------------------
    {"abc.mp4": {"label": "FAKE", "split": "train", "original": "def.mp4"},
     "def.mp4": {"label": "REAL", "split": "train"}}

Guarantees
----------
- A source group never spans two splits. Enforced by construction and re-checked
  before writing; a violation raises instead of writing a leaky manifest.
- Fake videos keep their `original_id` for provenance.
- Real videos with no derivatives form singleton groups.

Limitation (report this)
------------------------
DFDC metadata gives `original` linkage, not actor identity. Two different real
videos of the same person land in different groups, so this is source-group
disjoint but NOT guaranteed identity-disjoint.

CLI
---
    python -m src.data.build_manifest --config configs/dataset.yaml
    python -m src.data.build_manifest --metadata data/raw/train_sample/metadata.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from src.contracts import SPLITS, ManifestRow

__all__ = [
    "DEFAULT_TARGETS",
    "build_source_groups",
    "assign_splits",
    "build_manifest",
    "summarize",
    "write_manifest",
]

# plan 2.2 targets (per class)
DEFAULT_TARGETS: dict[str, dict[str, int]] = {
    "train": {"real": 525, "fake": 525},
    "val": {"real": 112, "fake": 112},
    "test": {"real": 113, "fake": 113},
}


def _normalise_label(raw: object) -> int:
    value = str(raw).strip().upper()
    if value in {"FAKE", "1", "TRUE"}:
        return 1
    if value in {"REAL", "0", "FALSE"}:
        return 0
    raise ValueError(f"build_manifest: unknown label {raw!r}, expected REAL/FAKE")


def build_source_groups(metadata: dict[str, dict[str, object]]) -> dict[str, str]:
    """Map video_id -> source_group_id by unioning fakes with their originals."""
    parent: dict[str, str] = {vid: vid for vid in metadata}

    def find(node: str) -> str:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for video_id, info in metadata.items():
        original = info.get("original")
        if isinstance(original, str) and original in parent:
            union(original, video_id)

    return {vid: find(vid) for vid in metadata}


def assign_splits(
    metadata: dict[str, dict[str, object]],
    targets: dict[str, dict[str, int]] | None = None,
    seed: int = 42,
) -> dict[str, str]:
    """Assign whole source groups to train/val/test, respecting per-class quotas.

    Greedy and deterministic: shuffle groups with the given seed, then fill each
    split's real/fake quota in the fixed order train -> val -> test so the larger
    partitions get first pick of scarce groups.
    """
    targets = targets or DEFAULT_TARGETS
    groups = build_source_groups(metadata)

    members: dict[str, list[str]] = defaultdict(list)
    for video_id, group in groups.items():
        members[group].append(video_id)
    for group in members:
        members[group].sort()

    def group_counts(group: str) -> dict[str, int]:
        counts = {"real": 0, "fake": 0}
        for vid in members[group]:
            counts["fake" if _normalise_label(metadata[vid]["label"]) == 1 else "real"] += 1
        return counts

    order = sorted(members)
    random.Random(seed).shuffle(order)

    remaining = {split: dict(quota) for split, quota in targets.items()}
    assignment: dict[str, str] = {}

    for group in order:
        counts = group_counts(group)
        if counts["real"] == 0 and counts["fake"] == 0:
            continue
        for split in SPLITS:
            quota = remaining[split]
            if counts["real"] <= quota["real"] and counts["fake"] <= quota["fake"]:
                assignment[group] = split
                quota["real"] -= counts["real"]
                quota["fake"] -= counts["fake"]
                break

    left = {s: q for s, q in remaining.items() if q["real"] > 0 or q["fake"] > 0}
    if left:
        raise ValueError(
            "assign_splits: quotas not filled "
            f"{left}. The metadata has too few eligible groups; lower the targets "
            "in configs/dataset.yaml instead of letting splits collapse."
        )

    return {vid: assignment[group] for vid, group in groups.items() if group in assignment}


def summarize(rows: list[ManifestRow]) -> dict[str, dict[str, int]]:
    """Per-split, per-class counts plus distinct group counts."""
    out: dict[str, dict[str, int]] = {
        split: {"real": 0, "fake": 0, "groups": 0, "total": 0} for split in SPLITS
    }
    groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        bucket = out[row.split]
        bucket["fake" if row.label == 1 else "real"] += 1
        bucket["total"] += 1
        groups[row.split].add(row.source_group_id)
    for split in SPLITS:
        out[split]["groups"] = len(groups[split])
    return out


def describe_summary(summary: dict[str, dict[str, int]]) -> str:
    lines = ["split summary:"]
    for split in SPLITS:
        s = summary[split]
        lines.append(
            f"  {split:5s} real={s['real']:5d} fake={s['fake']:5d} "
            f"total={s['total']:5d} groups={s['groups']:5d}"
        )
    total = sum(s["total"] for s in summary.values())
    lines.append(f"  total videos={total}")
    return "\n".join(lines)


def build_manifest(
    metadata: dict[str, dict[str, object]],
    video_root: str | Path,
    targets: dict[str, dict[str, int]] | None = None,
    seed: int = 42,
) -> list[ManifestRow]:
    """Build validated manifest rows and assert no source group spans splits."""
    video_root = Path(video_root)
    groups = build_source_groups(metadata)
    splits = assign_splits(metadata, targets=targets, seed=seed)

    rows: list[ManifestRow] = []
    for video_id, split in sorted(splits.items()):
        info = metadata[video_id]
        row = ManifestRow(
            video_id=video_id,
            filepath=str(video_root / video_id),
            label=_normalise_label(info["label"]),
            split=split,
            source_group_id=groups[video_id],
            original_id=str(info["original"]) if info.get("original") else None,
            clip_start_sec=0.0,
        ).validate()
        rows.append(row)

    seen: dict[str, str] = {}
    for row in rows:
        previous = seen.setdefault(row.source_group_id, row.split)
        if previous != row.split:
            raise ValueError(
                f"build_manifest: source group {row.source_group_id!r} spans "
                f"{previous} and {row.split} - refusing to write a leaky manifest"
            )

    print(describe_summary(summarize(rows)))
    return rows


def write_manifest(
    rows: list[ManifestRow],
    manifest_path: str | Path,
    summary_path: str | Path | None = None,
) -> Path:
    """Write manifest.csv and an optional split_summary.json."""
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "video_id",
        "filepath",
        "label",
        "original_id",
        "source_group_id",
        "split",
        "clip_start_sec",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "video_id": row.video_id,
                    "filepath": row.filepath,
                    "label": row.label,
                    "original_id": row.original_id or "",
                    "source_group_id": row.source_group_id,
                    "split": row.split,
                    "clip_start_sec": f"{row.clip_start_sec:.3f}",
                }
            )

    if summary_path is not None:
        summary_path = Path(summary_path)
        summary_path.write_text(
            json.dumps(
                {
                    "counts": summarize(rows),
                    "total": len(rows),
                    "limitation": "source-group disjoint, not identity-disjoint",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the DFDC subset manifest.")
    parser.add_argument("--metadata", default="data/raw/train_sample/metadata.json")
    parser.add_argument("--video-root", default="data/raw/train_sample")
    parser.add_argument("--out", default="data/manifests/manifest.csv")
    parser.add_argument("--summary", default="data/manifests/split_summary.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-quota",
        action="store_true",
        help="skip the 1500-video targets and accept every available video",
    )
    args = parser.parse_args(argv)

    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        print(f"[wait] metadata not found: {metadata_path}")
        print("       download DFDC first (see README quickstart step 1).")
        return 2

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    targets = None
    if args.no_quota:
        targets = {split: {"real": 10**9, "fake": 10**9} for split in SPLITS}

    rows = build_manifest(metadata, args.video_root, targets=targets, seed=args.seed)
    path = write_manifest(rows, args.out, args.summary)
    print(f"wrote {path} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

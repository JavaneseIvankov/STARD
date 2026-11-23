"""STAF-Net | src/data/build_manifest.py

Owner   : Person 1 + Person 2 (shared). Owner of record: Person 1.
Status  : IMPLEMENTED - two dataset layouts, verified on synthetic trees.

Layouts
-------
1. `dfdc`      Kaggle DFDC sample: flat directory + metadata.json
                   {"abc.mp4": {"label": "FAKE", "split": "train",
                                "original": "def.mp4"}}
2. `labeldirs` videos already separated by class:
                   <root>/real/*.mp4   -> label 0
                   <root>/fake/*.mp4   -> label 1
               with an OPTIONAL metadata.json / metadata.csv used only for
               `original` linkage (grouping) and provenance.

`--layout auto` picks labeldirs when `real/` or `fake/` exists, otherwise dfdc.

Guarantees
----------
- A source group never spans two splits. Re-checked before writing; a violation
  raises instead of writing a leaky manifest.
- The directory a video lives in is authoritative for its label; metadata can
  only add the `original` link. A mismatch is reported, not silently trusted.
- Videos that metadata claims but that are missing on disk are reported and
  skipped, so the manifest never points at a non-existent file.

Limitation (report this)
------------------------
DFDC metadata gives `original` linkage, not actor identity. Two different real
videos of the same person land in different groups, so this is source-group
disjoint but NOT guaranteed identity-disjoint.

CLI
---
    python -m src.data.build_manifest --video-root ../../datasets --layout labeldirs
    python -m src.data.build_manifest --video-root ../../datasets --no-quota
    python -m src.data.build_manifest --metadata data/raw/train_sample/metadata.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from src.contracts import SPLITS, ManifestRow

__all__ = [
    "DEFAULT_TARGETS",
    "VIDEO_EXTS",
    "build_source_groups",
    "assign_splits",
    "build_manifest",
    "build_manifest_from_dirs",
    "load_metadata_any",
    "scan_label_dirs",
    "summarize",
    "write_manifest",
]

# plan 2.2 targets (per class)
DEFAULT_TARGETS: dict[str, dict[str, int]] = {
    "train": {"real": 525, "fake": 525},
    "val": {"real": 112, "fake": 112},
    "test": {"real": 113, "fake": 113},
}

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}

LABEL_DIRS: dict[str, int] = {"real": 0, "fake": 1}


def _normalise_label(raw: object) -> int:
    value = str(raw).strip().upper()
    if value in {"FAKE", "1", "TRUE", "1.0"}:
        return 1
    if value in {"REAL", "0", "FALSE", "0.0"}:
        return 0
    raise ValueError(f"build_manifest: unknown label {raw!r}, expected REAL/FAKE/0/1")


# ---------------------------------------------------------------------------
# metadata readers
# ---------------------------------------------------------------------------
def load_metadata_any(path: str | Path | None) -> dict[str, dict[str, object]]:
    """Load metadata.json (DFDC dict or record list) or metadata.csv, tolerantly.

    Returns {video_id: {"label": int|None, "original": str|None, "relpath": str|None}}.
    Unknown columns are ignored; a missing file returns {} instead of failing.
    """
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        print(f"[info] metadata not found, continuing without it: {path}")
        return {}

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        with path.open(newline="", encoding="utf-8") as handle:
            payload = list(csv.DictReader(handle))

    out: dict[str, dict[str, object]] = {}

    def key_of(record: dict[str, object], fallback_index: int) -> str:
        for field in ("video_id", "filename", "file", "filepath", "path", "name"):
            value = record.get(field)
            if value:
                return Path(str(value)).name
        return f"__row{fallback_index}"

    if isinstance(payload, dict):
        for video_id, info in payload.items():
            info = info if isinstance(info, dict) else {}
            raw_label = info.get("label")
            out[Path(str(video_id)).name] = {
                "label": _normalise_label(raw_label) if raw_label is not None else None,
                "original": (Path(str(info["original"])).name if info.get("original") else None),
                "relpath": str(info.get("filepath") or info.get("path") or "") or None,
            }
        return out

    if isinstance(payload, list):
        for index, record in enumerate(payload):
            record = {str(k).lower(): v for k, v in dict(record).items()}
            raw_label = record.get("label") if record.get("label") not in (None, "") else None
            original = record.get("original") or record.get("original_id") or None
            out[key_of(record, index)] = {
                "label": _normalise_label(raw_label) if raw_label is not None else None,
                "original": Path(str(original)).name if original else None,
                "relpath": str(record.get("filepath") or record.get("path") or "") or None,
            }
        return out

    raise ValueError(f"load_metadata_any: unsupported metadata shape in {path}")


# ---------------------------------------------------------------------------
# directory scanning
# ---------------------------------------------------------------------------
def scan_label_dirs(root: str | Path) -> dict[str, tuple[Path, int]]:
    """Find videos under <root>/real and <root>/fake. Returns {name: (path, label)}."""
    root = Path(root)
    found: dict[str, tuple[Path, int]] = {}
    for dirname, label in LABEL_DIRS.items():
        base = root / dirname
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.suffix.lower() not in VIDEO_EXTS:
                continue
            if path.name in found:
                raise ValueError(
                    f"scan_label_dirs: duplicate filename {path.name!r} "
                    f"({found[path.name][0]} and {path})"
                )
            found[path.name] = (path, label)
    return found


# ---------------------------------------------------------------------------
# grouping + splitting
# ---------------------------------------------------------------------------
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


def _group_counts(
    metadata: dict[str, dict[str, object]], groups: dict[str, str]
) -> dict[str, dict[str, int]]:
    members: dict[str, list[str]] = defaultdict(list)
    for video_id, group in groups.items():
        members[group].append(video_id)
    out: dict[str, dict[str, int]] = {}
    for group, vids in members.items():
        counts = {"real": 0, "fake": 0}
        for vid in vids:
            counts["fake" if int(metadata[vid]["label"]) == 1 else "real"] += 1
        out[group] = counts
    return out


def targets_from_counts(
    metadata: dict[str, dict[str, object]],
    ratios: dict[str, float] | None = None,
) -> dict[str, dict[str, int]]:
    """Derive per-split, per-class quotas from the available pool."""
    ratios = ratios or {"train": 0.70, "val": 0.15, "test": 0.15}
    totals = Counter("fake" if int(info["label"]) == 1 else "real" for info in metadata.values())
    targets: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        targets[split] = {
            cls: int(round(totals[cls] * ratios.get(split, 0.0))) for cls in ("real", "fake")
        }
    return targets


def assign_splits(
    metadata: dict[str, dict[str, object]],
    targets: dict[str, dict[str, int]] | None = None,
    seed: int = 42,
    strict: bool = True,
) -> dict[str, str]:
    """Assign whole source groups to train/val/test, respecting per-class quotas.

    Greedy and deterministic: shuffle groups with the given seed, then fill each
    split's real/fake quota in the fixed order train -> val -> test.
    """
    targets = targets or DEFAULT_TARGETS
    groups = build_source_groups(metadata)
    counts = _group_counts(metadata, groups)

    order = sorted(counts)
    random.Random(seed).shuffle(order)

    remaining = {split: dict(quota) for split, quota in targets.items()}
    assignment: dict[str, str] = {}

    for group in order:
        group_count = counts[group]
        if group_count["real"] == 0 and group_count["fake"] == 0:
            continue
        for split in SPLITS:
            quota = remaining[split]
            if group_count["real"] <= quota["real"] and group_count["fake"] <= quota["fake"]:
                assignment[group] = split
                quota["real"] -= group_count["real"]
                quota["fake"] -= group_count["fake"]
                break

    left = {s: q for s, q in remaining.items() if q["real"] > 0 or q["fake"] > 0}
    if left and strict:
        raise ValueError(
            "assign_splits: quotas not filled "
            f"{left}. The pool has too few eligible groups; lower the targets or "
            "run with --no-quota to split by ratio instead."
        )

    return {vid: assignment[group] for vid, group in groups.items() if group in assignment}


# ---------------------------------------------------------------------------
# manifest assembly
# ---------------------------------------------------------------------------
def _rows_from_assignment(
    metadata: dict[str, dict[str, object]],
    splits: dict[str, str],
    paths: dict[str, Path],
) -> list[ManifestRow]:
    groups = build_source_groups(metadata)
    rows: list[ManifestRow] = []
    for video_id, split in sorted(splits.items()):
        info = metadata[video_id]
        rows.append(
            ManifestRow(
                video_id=video_id,
                filepath=str(paths[video_id]),
                label=int(info["label"]),
                split=split,
                source_group_id=groups[video_id],
                original_id=str(info["original"]) if info.get("original") else None,
                clip_start_sec=0.0,
            ).validate()
        )

    seen: dict[str, str] = {}
    for row in rows:
        previous = seen.setdefault(row.source_group_id, row.split)
        if previous != row.split:
            raise ValueError(
                f"build_manifest: source group {row.source_group_id!r} spans "
                f"{previous} and {row.split} - refusing to write a leaky manifest"
            )
    return rows


def build_manifest_from_dirs(
    video_root: str | Path,
    metadata_path: str | Path | None = None,
    targets: dict[str, dict[str, int]] | None = None,
    seed: int = 42,
    strict: bool = True,
) -> list[ManifestRow]:
    """Build a manifest from <root>/real and <root>/fake plus optional metadata."""
    video_root = Path(video_root)
    found = scan_label_dirs(video_root)
    if not found:
        raise FileNotFoundError(
            f"build_manifest_from_dirs: no videos under {video_root}/real or {video_root}/fake"
        )

    meta = load_metadata_any(metadata_path)
    metadata: dict[str, dict[str, object]] = {}
    paths: dict[str, Path] = {}
    mismatches = 0

    for name, (path, dir_label) in found.items():
        info = meta.get(name, {})
        meta_label = info.get("label")
        if meta_label is not None and int(meta_label) != dir_label:
            mismatches += 1
        metadata[name] = {
            "label": dir_label,  # directory wins
            "original": info.get("original") if info.get("original") in found else None,
        }
        paths[name] = path

    orphan_originals = sum(1 for info in meta.values() if info.get("original") not in (None, "") and info["original"] not in found)
    missing_on_disk = len(set(meta) - set(found))

    print(
        f"[info] scanned {len(found)} videos "
        f"(real={sum(1 for _, l in found.values() if l == 0)}, "
        f"fake={sum(1 for _, l in found.values() if l == 1)})"
    )
    if mismatches:
        print(f"[warn] {mismatches} videos had a metadata label that disagreed with their directory; directory used")
    if orphan_originals:
        print(f"[warn] {orphan_originals} metadata rows reference an original that is not in this subset")
    if missing_on_disk:
        print(f"[warn] {missing_on_disk} metadata rows have no matching video on disk (ignored)")

    if targets is None and not strict:
        targets = targets_from_counts(metadata)

    splits = assign_splits(metadata, targets=targets, seed=seed, strict=strict)
    rows = _rows_from_assignment(metadata, splits, paths)
    print(describe_summary(summarize(rows)))
    return rows


def build_manifest(
    metadata: dict[str, dict[str, object]],
    video_root: str | Path,
    targets: dict[str, dict[str, int]] | None = None,
    seed: int = 42,
) -> list[ManifestRow]:
    """Build manifest rows for a flat DFDC layout (videos directly under root)."""
    video_root = Path(video_root)
    prepared: dict[str, dict[str, object]] = {}
    paths: dict[str, Path] = {}
    for video_id, info in metadata.items():
        if info.get("label") is None:
            continue
        path = video_root / video_id
        if not path.exists():
            print(f"[warn] missing on disk, skipped: {path}")
            continue
        prepared[video_id] = {"label": int(info["label"]), "original": info.get("original")}
        paths[video_id] = path

    if not prepared:
        raise FileNotFoundError(f"build_manifest: no videos found under {video_root}")

    splits = assign_splits(prepared, targets=targets, seed=seed)
    rows = _rows_from_assignment(prepared, splits, paths)
    print(describe_summary(summarize(rows)))
    return rows


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------
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
        summary_path.parent.mkdir(parents=True, exist_ok=True)
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


def _detect_layout(video_root: Path) -> str:
    if (video_root / "real").is_dir() or (video_root / "fake").is_dir():
        return "labeldirs"
    return "dfdc"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the DFDC subset manifest.")
    parser.add_argument("--video-root", default="data/raw/train_sample")
    parser.add_argument("--metadata", default=None, help="metadata.json / metadata.csv (optional for labeldirs)")
    parser.add_argument("--layout", default="auto", choices=("auto", "dfdc", "labeldirs"))
    parser.add_argument("--out", default="data/manifests/manifest.csv")
    parser.add_argument("--summary", default="data/manifests/split_summary.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-quota",
        action="store_true",
        help="split the available pool by ratio (70/15/15) instead of fixed 1500-video quotas",
    )
    args = parser.parse_args(argv)

    video_root = Path(args.video_root)
    if not video_root.exists():
        print(f"[wait] video root not found: {video_root}")
        return 2

    layout = _detect_layout(video_root) if args.layout == "auto" else args.layout
    print(f"[info] layout={layout} root={video_root}")

    if layout == "labeldirs":
        metadata = args.metadata
        if metadata is None:
            for candidate in ("metadata.json", "metadata.csv"):
                if (video_root / candidate).exists():
                    metadata = str(video_root / candidate)
                    break
        rows = build_manifest_from_dirs(
            video_root,
            metadata_path=metadata,
            targets=None if args.no_quota else DEFAULT_TARGETS,
            seed=args.seed,
            strict=not args.no_quota,
        )
    else:
        metadata_path = args.metadata or str(video_root / "metadata.json")
        if not Path(metadata_path).exists():
            print(f"[wait] metadata not found: {metadata_path}")
            return 2
        meta = load_metadata_any(metadata_path)
        rows = build_manifest(
            meta,
            video_root,
            targets=None if args.no_quota else DEFAULT_TARGETS,
            seed=args.seed,
        )

    path = write_manifest(rows, args.out, args.summary)
    print(f"wrote {path} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

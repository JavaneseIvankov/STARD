# Cached backbone features — format spec v0.1

Owner: Orang A (DINOv3) + Orang B (V-JEPA), consumed by Orang 1 (`src/data/dataset.py`).

Purpose: backbones are frozen, so extract once with `scripts/extract_features.py` and
reuse the embeddings for every fusion/head training run. This is what makes the
full pipeline trainable on a single GPU (README §7.3).

## Layout

```text
datasets/cached_features/
├── index.csv                 # one row per video, all splits
├── train/shard-00000.pt      # ~500 videos per shard
├── train/shard-00001.pt
├── val/shard-00000.pt
└── test/shard-00000.pt
```

Sharding matters: one file per video means ~100k tiny files and a slow
`DataLoader` on network filesystems.

## `index.csv`

| column | type | notes |
| :--- | :--- | :--- |
| `video_id` | str | stable key, matches `datasets/processed/` |
| `identity_id` | str | identity-level split source of truth |
| `split` | str | `train` / `val` / `test` |
| `shard` | str | e.g. `train/shard-00000.pt` |
| `shard_index` | int | position inside `records` list |
| `label_cls` | int | 0 real / 1 deepfake / 2 physical_spoof |
| `num_frames` | int | 16 |
| `has_spatial_mask` | 0/1 | whether `label_spatial` exists |
| `has_temporal_mask` | 0/1 | whether `label_temporal` exists |
| `source_sha` | str | sha256 of the source video, dedup + provenance |

## Shard payload

`torch.save` of a dict (`_use_new_zipfile_serialization=True`):

```python
{
    "format_version": "0.1",
    "backbone": {"spatial": "dinov3-vits16", "temporal": "vjepa2.1-vitb16"},
    "records": [
        {
            "video_id": "...",
            "spatial": torch.Tensor,   # (16, 196, 384)
            "temporal": torch.Tensor,  # (16, 196, 768)
            "label_cls": int,
            "label_spatial": torch.Tensor | None,   # (16, 224, 224)
            "label_temporal": torch.Tensor | None,  # (16,)
        },
        # ...
    ],
}
```

A unimodal baseline may omit the branch it does not use (`"temporal": None`).
`src/data/dataset.py` must still yield a valid `src.contracts.BatchInput`.

## Dtype policy

- Store `spatial` / `temporal` as **float16** — halves disk and load time, no
  measurable accuracy effect because backbones are frozen.
- Cast to **float32** in `dataset.py` (`tensor.float()`) so fusion and heads train
  in fp32 / AMP bf16 without silent precision surprises.
- Never store post-normalization features twice; normalization already happened
  inside the backbone wrapper.

## Invariants

1. `spatial.shape == (16, 196, 384)` and `temporal.shape == (16, 196, 768)` —
   enforced by `src/contracts.py`.
2. `format_version` mismatch on load = hard error, not a warning.
3. A change to backbone weights, patch size, input size, or frame sampling
   invalidates the whole cache directory. Write a `MANIFEST.json` per cache run:

```json
{
  "format_version": "0.1",
  "backbones": {"spatial": "dinov3-vits16", "temporal": "vjepa2.1-vitb16"},
  "image_size": 224,
  "num_frames": 16,
  "sampling": "uniform",
  "created_at": "...",
  "git_commit": "..."
}
```

4. Extraction runs under `torch.no_grad()` with `model.eval()`.
5. Masks are stored per video, not per batch; collation is the Dataset's job.

## Disk budget

Per video, fp16: `16 * 196 * (384 + 768) * 2 bytes ≈ 7.2 MiB`.
100k videos ≈ 720 GB — plan shards and pruning before a full extraction run.
Extract spatial-only or temporal-only first if disk is tight.

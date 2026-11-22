# Cached backbone features — format spec v0.2

Owner: Person 1 + Person 2 (producers), Person 3 (consumer).

Stage II trains only on cached per-video embeddings, so the cache is the contract
between Stage I and the ablation matrix. Backbones are expensive; the cache is not.

## Layout

```text
data/features/
├── dino_frozen/
│   ├── MANIFEST.json
│   ├── index.csv
│   └── train/shard-00000.pt
├── dino_lora/
├── vjepa_frozen/
└── vjepa_lora/
```

One directory per variant. Splits are subdirectories; shards hold ~500 videos
each so the DataLoader does not open thousands of tiny files.

## `index.csv`

| column | type | notes |
| :--- | :--- | :--- |
| `video_id` | str | matches `data/manifests/manifest.csv` |
| `source_group_id` | str | grouping key for bootstrap CI and split checks |
| `split` | str | `train` / `val` / `test` |
| `shard` | str | e.g. `train/shard-00000.pt` |
| `shard_index` | int | position inside the shard's `records` list |
| `label` | int | 0 real, 1 fake |
| `dim` | int | 384 for dino variants, 768 for vjepa variants |

## Shard payload

`torch.save` of a dict, `_use_new_zipfile_serialization=True`:

```python
{
    "format_version": "0.2",
    "variant": "dino_lora",
    "checkpoint_id": "dinov3-vits16-lvd1689m+lora-r8-a16",
    "records": [
        {
            "video_id": "abc123.mp4",
            "label": 1,
            "z": torch.Tensor,          # (384,) dino  |  (768,) vjepa
            "source_group_id": "def.mp4",
        },
    ],
}
```

One pooled vector per video. Not a token grid: Stage II never sees spatial or
temporal tokens, which is why 1,500 videos × 4 variants is ~13 MB fp32.

## Dtype policy

Store and train in **float32**. A 384-dim vector is 1.5 KiB, so fp16 would save
noise-level space while adding cast bugs at the projection layer. Cast only if
the subset grows by orders of magnitude.

## Invariants

1. `z.shape == (384,)` for `dino_*`, `(768,)` for `vjepa_*` — enforced by
   `src.contracts.FeatureRecord`.
2. `format_version` mismatch on load is a hard error, not a warning.
3. Variants are never mixed silently: a run records exactly which variants it read.
4. Changing backbone weights, LoRA config, frame indices, resolution, pooling,
   or clip sampling invalidates the cache. `MANIFEST.json` records all of them:

```json
{
  "format_version": "0.2",
  "variant": "vjepa_lora",
  "backbone": "vjepa2.1-vitb16",
  "checkpoint_id": "...",
  "preprocessing_version": "clips-v1",
  "num_frames": 16,
  "resolution": 384,
  "pooling": "mean",
  "created_at": "...",
  "git_commit": "..."
}
```

5. Extraction runs under `torch.no_grad()` with `model.eval()`.
6. Shard writing is resume-safe: an existing shard with a matching
   `format_version` and `checkpoint_id` is skipped, never silently overwritten.
7. Frozen and LoRA variants of the same branch must be produced from the same
   clip cache and the same manifest, otherwise A1 vs A3 is not a fair comparison.

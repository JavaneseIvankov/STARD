# datasets (git-ignored)

Local-only data layout. Nothing here is committed.

- `raw/` — original videos
- `processed/` — 16-frame facial crops (224x224) + identity-level split index (.csv)
- `cached_features/` — pre-extracted DINOv3 / V-JEPA embeddings (.pt)

Status: PLACEHOLDER — populated by `scripts/run_preprocess.py` and `scripts/extract_features.py`.

# data/

Not committed (see .gitignore). Populated by the pipeline:

- `images/`          — collected images (data_collection/collect.py)
- `manifest.jsonl`   — image records + weak labels
- `split.json`       — 70/30 index-eval / train split (eval/labels.py)

Generated evaluation files land in `eval/` (labels.json, swap_pairs.json).
Embeddings and rankings land in `artifacts/`.

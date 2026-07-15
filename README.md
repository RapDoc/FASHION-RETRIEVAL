# Multimodal Fashion & Context Retrieval

A retrieval system that searches a fashion image database from natural-language
descriptions, with a focus on **compositional colour–garment binding** — the case
where vanilla CLIP fails (distinguishing "red tie + white shirt" from "white tie
+ red shirt").

Full write-up: [`docs/writeup.pdf`](docs/writeup.pdf)

## Key result

| System | narrow R@5 | broad P@5 | binding acc | colour-critical |
|---|---|---|---|---|
| CLIP ViT-B/32 | 0.225 | 0.533 | 0.541 | 0.509 |
| FashionSigLIP (global) | 0.500 | 0.700 | 0.674 | 0.661 |
| + region binding | 0.500 | 0.650 | 0.730 | 0.714 |
| **+ cross-encoder (final)** | 0.500 | 0.683 | **0.730** | **0.714** |

Binding accuracy (correct vs colour-swapped image; chance = 0.500) improves from
**0.541 → 0.730**. On colour-critical pairs, where colour is the only cue, from
**0.509 (chance) → 0.714**.

## Architecture

Two workflows, separated as the brief requires:

- **Indexer** (`indexer/`, offline, GPU): SegFormer garment masks → region
  embeddings + masked LAB colour + a global embedding per image.
- **Retriever** (`retriever/`, CPU): global ANN recall → Hungarian binding
  rerank → BLIP-ITM cross-encoder verification on the top-20 → score fusion.

All heavy models run offline; the retriever loads precomputed artifacts and runs
without a GPU.

## Data

The dataset (~1,200 images spanning the environment × garment × colour × formality
grid) is **not committed** to keep the repository light. It is fully reproducible:

```bash
export PEXELS_API_KEY=...        # free key from pexels.com/api
python data_collection/collect.py --provider pexels --per-cell 8
python eval/labels.py            # generates labels + splits from the collected images
```

Every image is retrieved by a known grid cell, which is stored as its weak label,
so evaluation labels are generated rather than hand-annotated.

The exact image set used for the results in the write-up is available as a Kaggle
dataset: **`<add your public Kaggle dataset link here>`**. Download it into
`data/` to reproduce the reported metrics exactly.

Scripts fail with a clear message (not a stack trace) if the dataset or index is
missing, pointing back to the relevant step.

## Quickstart

```bash
pip install -r requirements.txt

# 1. Build the dataset (see data_collection/) or drop images into data/images/
python data_collection/collect.py --provider pexels --per-cell 8

# 2. Generate evaluation labels
python eval/labels.py

# 3. Run the full pipeline (indexing needs GPU; rest is CPU)
bash scripts/run_pipeline.sh
```

Single query (prints ranked image IDs; `--show N` also saves a top-N image grid):

```bash
python retriever/search_two_stage.py --encoder fashionsiglip \
    --query "a red tie and a white shirt in a formal setting" --show 5
```

Visual results for all five official queries in one image:

```bash
python eval/show_results.py        # -> artifacts/official_queries_topk.jpg
```

## Repository layout

```
configs/            # vocab (garments, colours, scenes) + scoring weights — all tunables, no magic numbers in code
data_collection/    # grid-based image collector; weak labels from retrieval query
indexer/            # embed_global.py, embed_regions.py  (GPU, offline)
retriever/          # parse_query · search_global · search_two_stage · binding · rerank_crossencoder  (CPU)
eval/               # queries.json, labels.py, run_eval.py, show_results.py
scripts/            # run_pipeline.sh
docs/               # writeup.pdf, writeup.tex
```

## Evaluation design

- **Labels are generated, not hand-annotated**: each query is a predicate over
  image metadata (`eval/labels.py`), so 30 queries × 863 images is a one-command job.
- **Binding accuracy** is measured on deliberately-planted colour-swap pairs
  (`eval/swap_pairs.json`, generated) — the metric that isolates binding.
- **Metrics per query type**: R@5 for narrow attribute queries, P@5 for broad
  scene/style queries (~20% of the corpus is relevant, so recall is uninformative),
  binding accuracy for compositional queries.

## Notes

- Query parsing uses a regex/keyword fallback by default (no API key needed);
  set `GROQ_API_KEY` to use an LLM parser instead.
- A 70/30 index/train split is enforced (`data/split.json`) so the held-out set
  is never trained on — relevant if the encoder is later fine-tuned.

"""Stage-1 retrieval: ANN over the single global vector per image.

Used on its own as the CLIP/FashionSigLIP baseline, and as the recall stage that
feeds the binding reranker. Recall is O(log N) in the corpus size; the expensive
reranking only ever runs on the returned shortlist.

    python retriever/search_global.py --encoder clip
    python retriever/search_global.py --encoder clip --query "a red tie and a white shirt"
"""

from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
import sys; sys.path.insert(0, str(ROOT))
from indexer.embed_global import load_encoder


def build_qdrant(V: np.ndarray, ids: list[str]):
    """Qdrant in local mode — no server, no docker. The brief says spend the
    time on ML logic, not on index engineering, so we use the boring option."""
    from qdrant_client import QdrantClient, models
    c = QdrantClient(":memory:")
    c.create_collection("imgs", vectors_config=models.VectorParams(
        size=V.shape[1], distance=models.Distance.COSINE))
    c.upsert("imgs", points=[models.PointStruct(id=i, vector=V[i].tolist(),
                                                payload={"image_id": ids[i]})
                             for i in range(len(ids))])
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="clip")
    ap.add_argument("--query", help="ad-hoc query; omit to run the full eval set")
    ap.add_argument("--topk", type=int, default=200)
    a = ap.parse_args()

    V = np.load(ART / f"global_{a.encoder}.npy")
    ids = json.loads((ART / f"index_{a.encoder}.json").read_text())["image_ids"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, enc_txt = load_encoder(a.encoder, device)
    client = build_qdrant(V, ids)

    def search(text: str, k: int) -> list[str]:
        q = enc_txt([text])[0]
        hits = client.query_points("imgs", query=q.tolist(), limit=k).points
        return [h.payload["image_id"] for h in hits]

    if a.query:
        for r, iid in enumerate(search(a.query, 10), 1):
            print(f"{r:2d}. {iid}")
        return

    qs = json.loads((ROOT / "eval" / "queries.json").read_text())["queries"]
    rankings = {q["id"]: search(q["text"], a.topk) for q in qs}
    out = ART / f"rankings_{a.encoder}_global.json"
    out.write_text(json.dumps(rankings))
    print(f"ranked {len(qs)} queries over {len(ids)} images -> {out.name}")


if __name__ == "__main__":
    main()

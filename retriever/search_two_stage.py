"""Two-stage retriever: global recall then binding-aware rerank.

Stage 1 is a global-vector ANN over the whole corpus (O(log N)); stage 2 reranks
the ~200 recalled candidates with the Hungarian binding score fused with global
and scene similarity. The expensive logic only ever touches the shortlist, which
is what keeps the system scalable.

    python retriever/search_two_stage.py --encoder fashionsiglip
    python retriever/search_two_stage.py --encoder fashionsiglip --query "a red tie and a white shirt"
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
sys.path.insert(0, str(ROOT))
from indexer.embed_global import load_encoder
from retriever.parse_query import parse
from retriever.binding import binding_score

CFG = yaml.safe_load((ROOT / "configs" / "scoring.yaml").read_text()) \
    if (ROOT / "configs" / "scoring.yaml").exists() else {}
W_BIND = CFG.get("w_bind", 0.55)
W_GLOBAL = CFG.get("w_global", 0.30)
W_SCENE = CFG.get("w_scene", 0.15)
RECALL_R = CFG.get("recall_r", 200)


def _norm(v):
    return v / (np.linalg.norm(v) + 1e-9)


def save_grid(query, ids, root):
    """Save the top results as a single image grid for quick visual inspection."""
    import json as _json
    from PIL import Image
    path = {_json.loads(l)["image_id"]: _json.loads(l)["path"]
            for l in (root / "data" / "manifest.jsonl").read_text().splitlines() if l.strip()}
    imgs = []
    for iid in ids:
        try:
            imgs.append(Image.open(root / path[iid]).convert("RGB").resize((224, 300)))
        except Exception:
            pass
    if not imgs:
        return
    grid = Image.new("RGB", (224 * len(imgs), 300), "white")
    for i, im in enumerate(imgs):
        grid.paste(im, (224 * i, 0))
    out = root / "artifacts" / ("result_" + "_".join(query.split()[:4]) + ".jpg")
    out.parent.mkdir(exist_ok=True)
    grid.save(out)
    print(f"\nsaved top-{len(imgs)} grid -> {out.relative_to(root)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="fashionsiglip")
    ap.add_argument("--query")
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--use-groq", action="store_true")
    ap.add_argument("--show", type=int, default=0,
                    help="save the top-N results as an image grid for inspection")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not (ART / f"global_{a.encoder}.npy").exists():
        raise SystemExit("No index found. Run indexer/embed_global.py and "
                         "indexer/embed_regions.py first (see README).")

    # Stage-1 index: one global vector per image.
    G = np.load(ART / f"global_{a.encoder}.npy")
    gids = json.loads((ART / f"index_{a.encoder}.json").read_text())["image_ids"]
    gidx = {iid: i for i, iid in enumerate(gids)}

    # Stage-2 index: per-image garment regions and their embeddings.
    rregs = [json.loads(l) for l in (ART / f"regions_{a.encoder}.jsonl").read_text().splitlines() if l.strip()]
    region_emb = np.load(ART / f"region_emb_{a.encoder}.npy")
    regions_by_img = {r["image_id"]: r["regions"] for r in rregs}

    enc_img, enc_txt = load_encoder(a.encoder, device)

    def garment_text_embs(intent):
        """Embed the "a {colour} {type}" phrase for each queried garment."""
        keys = {f"a {g.get('color') or ''} {g['type']}".replace("  ", " ").strip()
                for g in intent["garments"]}
        if not keys:
            return {}
        vecs = enc_txt(list(keys))
        return {k: _norm(vecs[i]) for i, k in enumerate(keys)}

    def search(text: str, k: int):
        intent = parse(text, use_groq=a.use_groq)
        qvec = _norm(enc_txt([text])[0])

        # Stage 1: global cosine recall.
        sims = G @ qvec
        cand = [(gids[i], float(sims[i])) for i in np.argsort(-sims)[:RECALL_R]]

        # Stage 2: rerank the shortlist.
        gte = garment_text_embs(intent)
        scene_vec = _norm(enc_txt([f"a photo in a {intent['scene']}"])[0]) if intent.get("scene") else None

        rescored = []
        for iid, gsim in cand:
            regs = regions_by_img.get(iid, [])
            if intent["garments"]:
                bind, coverage = binding_score(intent["garments"], regs, region_emb, gte)
                # Binding only helps when there are 2+ garments to disambiguate.
                # For a single-garment query global recall is already strong, so we
                # give binding a reduced weight and hand the rest back to global.
                comp_factor = 0.4 if len(intent["garments"]) == 1 else 1.0
                eff_bind_w = W_BIND * coverage * comp_factor
                rest = (W_BIND - eff_bind_w) + W_GLOBAL
                scene_sim = float(G[gidx[iid]] @ scene_vec) if scene_vec is not None else 0.0
                score = eff_bind_w * bind + rest * gsim + W_SCENE * scene_sim
            else:
                # No garment named: lean on global recall and scene similarity.
                scene_sim = float(G[gidx[iid]] @ scene_vec) if scene_vec is not None else gsim
                score = W_GLOBAL * gsim + (W_BIND + W_SCENE) * scene_sim
            rescored.append((iid, score))
        rescored.sort(key=lambda x: -x[1])
        return [iid for iid, _ in rescored[:k]], intent

    if a.query:
        ids, intent = search(a.query, 10)
        gs = ", ".join(f"{g['color'] or '?'}:{g['type']}" for g in intent["garments"]) or "(none)"
        print(f"intent: [{gs}] scene={intent['scene']} formality={intent['formality']}\n")
        for r, iid in enumerate(ids, 1):
            print(f"{r:2d}. {iid}")
        if a.show:
            save_grid(a.query, ids[:a.show], ROOT)
        return

    qs = json.loads((ROOT / "eval" / "queries.json").read_text())["queries"]
    rankings = {}
    for q in qs:
        ids, _ = search(q["text"], a.topk)
        rankings[q["id"]] = ids
        print(f"  {q['id']:9s} done", end="\r")
    out = ART / f"rankings_{a.encoder}_twostage.json"
    out.write_text(json.dumps(rankings))
    print(f"\nranked {len(qs)} queries (two-stage) -> {out.name}")


if __name__ == "__main__":
    main()

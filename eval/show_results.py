"""Build a labelled top-5 image grid for each official query.

Reads a rankings file (default: the full two-stage + cross-encoder run) and, for
each of the five official queries, writes a row of its top-5 images with the query
text as a caption. Handy for eyeballing retrieval quality and for the write-up.

    python eval/show_results.py
    python eval/show_results.py --rankings artifacts/rankings_fashionsiglip_twostage.json
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
CELL = (224, 300)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rankings", default=str(ART / "rankings_fashionsiglip_crossencoder.json"))
    ap.add_argument("--topk", type=int, default=5)
    a = ap.parse_args()

    rankings = json.loads(Path(a.rankings).read_text())
    queries = json.loads((ROOT / "eval" / "queries.json").read_text())["queries"]
    official = [q for q in queries if q.get("official")]
    path = {json.loads(l)["image_id"]: json.loads(l)["path"]
            for l in (ROOT / "data" / "manifest.jsonl").read_text().splitlines() if l.strip()}

    cap_h = 34
    rows = []
    for q in official:
        ids = rankings.get(q["id"], [])[:a.topk]
        row = Image.new("RGB", (CELL[0] * a.topk, CELL[1] + cap_h), "white")
        ImageDraw.Draw(row).text((6, 9), f'{q["id"]}: {q["text"]}', fill="black")
        for i, iid in enumerate(ids):
            try:
                im = Image.open(ROOT / path[iid]).convert("RGB").resize(CELL)
                row.paste(im, (CELL[0] * i, cap_h))
            except Exception:
                pass
        rows.append(row)

    grid = Image.new("RGB", (CELL[0] * a.topk, (CELL[1] + cap_h) * len(rows)), "white")
    for i, r in enumerate(rows):
        grid.paste(r, (0, (CELL[1] + cap_h) * i))
    out = ART / "official_queries_topk.jpg"
    grid.save(out)
    print(f"saved {len(rows)} query rows -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

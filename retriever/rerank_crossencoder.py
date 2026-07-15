"""Stage-3 cross-encoder verification with BLIP-ITM.

Stages 1-2 embed image and text separately. A cross-encoder instead runs the
image and query through joint attention, scoring the whole proposition at once,
which helps the compositional cases. It is expensive per pair, so it only reranks
the top-K candidates from stage 2 and fuses its score with the stage-2 order.

    python retriever/rerank_crossencoder.py --encoder fashionsiglip --topk_ce 20
"""

from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
CFG = yaml.safe_load((ROOT / "configs" / "scoring.yaml").read_text())
W_CE = CFG.get("w_crossencoder", 0.5)
TOPK_CE = CFG.get("topk_crossencoder", 20)

BLIP_ID = "Salesforce/blip-itm-base-coco"


def load_itm(device):
    from transformers import BlipProcessor, BlipForImageTextRetrieval
    proc = BlipProcessor.from_pretrained(BLIP_ID)
    model = BlipForImageTextRetrieval.from_pretrained(BLIP_ID).to(device).eval()

    def itm_prob(img: Image.Image, text: str) -> float:
        x = proc(images=img, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**x)                       # itm_score logits [neg, pos]
        return float(torch.softmax(out.itm_score, dim=1)[0, 1].cpu())

    return itm_prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="fashionsiglip")
    ap.add_argument("--topk_ce", type=int, default=TOPK_CE)
    ap.add_argument("--stage2", default=None,
                    help="stage-2 rankings json (default: rankings_<enc>_twostage.json)")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    stage2_path = Path(a.stage2) if a.stage2 else ART / f"rankings_{a.encoder}_twostage.json"
    if not stage2_path.exists():
        raise SystemExit("No two-stage rankings found. Run "
                         "retriever/search_two_stage.py first (see README).")
    stage2 = json.loads(stage2_path.read_text())
    queries = {q["id"]: q for q in json.loads((ROOT / "eval" / "queries.json").read_text())["queries"]}

    # path lookup
    manifest = [json.loads(l) for l in (ROOT / "data" / "manifest.jsonl").read_text().splitlines() if l.strip()]
    path_of = {}
    for r in manifest:
        path_of.setdefault(r["image_id"], r["path"])

    itm_prob = load_itm(device)
    out = {}
    for qid, ranked in stage2.items():
        text = queries[qid]["text"]
        head, tail = ranked[:a.topk_ce], ranked[a.topk_ce:]
        # stage-2 order gives a descending pseudo-score for the head; normalise 0..1
        s2 = {iid: 1.0 - i / max(len(head), 1) for i, iid in enumerate(head)}
        rescored = []
        for iid in head:
            try:
                img = Image.open(ROOT / path_of[iid]).convert("RGB")
            except Exception:
                rescored.append((iid, s2[iid])); continue
            p = itm_prob(img, text)
            rescored.append((iid, W_CE * p + (1 - W_CE) * s2[iid]))
        rescored.sort(key=lambda x: -x[1])
        out[qid] = [iid for iid, _ in rescored] + tail   # reranked head + untouched tail
        print(f"  {qid:9s} CE-reranked top-{len(head)}", end="\r")

    dst = ART / f"rankings_{a.encoder}_crossencoder.json"
    dst.write_text(json.dumps(out))
    print(f"\ncross-encoder rerank -> {dst.name}")


if __name__ == "__main__":
    main()

"""Generate relevance labels from image metadata (no hand annotation).

Each image carries the grid cell that retrieved it (its weak label) or
Fashionpedia ground truth. A query is a predicate over that metadata, so
evaluating it yields a graded relevance label (0/1/2). Also builds the
correct-vs-swapped image pairs used for binding accuracy, and the index/train
split.

    python eval/labels.py            # after collect.py
    python eval/labels.py --selftest # runs on synthetic data, no dataset needed
"""

from __future__ import annotations

import argparse, json, random
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
VOCAB = yaml.safe_load((ROOT / "configs" / "vocab.yaml").read_text())
QUERIES = json.loads((ROOT / "eval" / "queries.json").read_text())["queries"]

REL_STRONG, REL_WEAK, REL_NONE = 2, 1, 0

# ---- alias -> canonical -----------------------------------------------------
def _amap(section: str) -> dict[str, str]:
    m = {}
    for canon, meta in VOCAB[section].items():
        m[canon] = canon
        for a in (meta or {}).get("aliases", []) or []:
            m[a.lower()] = canon
    return m

GARMENT_A, COLOR_A, ENV_A, FORM_A = (_amap(s) for s in
                                     ("garments", "colors", "environments", "formality"))


def canon(v, table):
    return table.get(str(v).lower()) if v else None


# ---------------------------------------------------------------------------
# 1. FACTS - normalise heterogeneous sources into one shape
# ---------------------------------------------------------------------------
def facts_from_web(rec: dict) -> dict:
    """Weak labels: the grid cell that retrieved the image."""
    w = rec["weak"]
    gs = []
    if w.get("garment"):
        gs.append({"type": canon(w["garment"], GARMENT_A), "color": canon(w.get("color"), COLOR_A)})
    if w.get("second_garment"):                      # the 2-garment Q5/swap cells
        sg = w["second_garment"]
        gs.append({"type": canon(sg["type"], GARMENT_A), "color": canon(sg.get("color"), COLOR_A)})
    return {"image_id": rec["image_id"], "source": rec["source"], "garments": gs,
            "scene": canon(w.get("environment"), ENV_A),
            "formality": canon(w.get("formality"), FORM_A),
            "label_conf": "weak"}


def facts_from_fashionpedia(rec: dict) -> dict:
    """Fashionpedia ships garment category + colour attributes = STRONG labels."""
    gs = [{"type": canon(g.get("category"), GARMENT_A), "color": canon(g.get("color"), COLOR_A)}
          for g in rec.get("garments", [])]
    gs = [g for g in gs if g["type"]]
    forms = [VOCAB["garments"][g["type"]]["formality_prior"] for g in gs if g["type"] in VOCAB["garments"]]
    return {"image_id": rec["image_id"], "source": "fashionpedia", "garments": gs,
            "scene": canon(rec.get("scene"), ENV_A),          # usually None -> studio
            "formality": max(set(forms), key=forms.count) if forms else None,
            "label_conf": "strong"}


# ---------------------------------------------------------------------------
# 2. PREDICATE EVALUATION
# ---------------------------------------------------------------------------
def _garment_hit(need: dict, have: list[dict], used: set[int]) -> int | None:
    """One-to-one: a region already consumed by another clause cannot be reused.
    This is the labelling-side mirror of the Hungarian matcher in the retriever —
    without it, 'red tie + white shirt' would be satisfied by a single red shirt."""
    types = need["type"] if isinstance(need.get("type"), list) else ([need["type"]] if need.get("type") else None)
    for i, g in enumerate(have):
        if i in used:
            continue
        if types and g["type"] not in types:
            continue
        if need.get("color") and g["color"] != need["color"]:
            continue
        return i
    return None


def relevance(pred: dict, f: dict) -> int:
    need = pred.get("garments", [])
    used, matched = set(), 0
    for n in need:
        i = _garment_hit(n, f["garments"], used)
        if i is not None:
            used.add(i); matched += 1

    scene_ok = "scene_any" not in pred or f["scene"] in pred["scene_any"]
    form_ok = "formality_any" not in pred or f["formality"] in pred["formality_any"]

    if not need:                                    # pure scene/vibe query (S*, V*)
        return REL_STRONG if (scene_ok and form_ok) else REL_NONE

    if matched == len(need):
        return REL_STRONG if (scene_ok and form_ok) else REL_WEAK
    if matched >= 1 and matched >= len(need) / 2 and scene_ok:
        return REL_WEAK                             # partial: garment right, binding unproven
    return REL_NONE


# ---------------------------------------------------------------------------
# 3. SWAP PAIRS -> BINDING ACCURACY
# ---------------------------------------------------------------------------
def build_swap_pairs(all_facts: list[dict]) -> list[dict]:
    """
    METRIC DEFINITION (this is the number the whole report hangs on):

      For query q with correct binding, pick image A (satisfies q) and image B
      (satisfies q's colour-SWAPPED twin). A and B contain the SAME garments and
      the SAME colours — only the binding differs. The system scores 1 iff
      rank(A) < rank(B).  Chance = 50%.

    Isolating binding from every other confound is exactly why a global-vector
    model scores ~chance here while looking fine on R@5.
    """
    byid = {q["id"]: q for q in QUERIES}
    pairs = []
    for q in QUERIES:
        if not q.get("swap_of"):
            continue
        orig = byid[q["swap_of"]]
        A = [f["image_id"] for f in all_facts if relevance(orig["predicate"], f) == REL_STRONG]
        B = [f["image_id"] for f in all_facts if relevance(q["predicate"], f) == REL_STRONG]
        A, B = set(A) - set(B), set(B) - set(A)          # unambiguous only
        if A and B:
            pairs.append({"query_id": orig["id"], "query_text": orig["text"],
                          "difficulty": orig.get("difficulty", "unspecified"),
                          "swap_query_text": q["text"],
                          "positive_ids": sorted(A), "swapped_negative_ids": sorted(B)})
    return pairs


# ---------------------------------------------------------------------------
# 4. SPLIT - the leak guard
# ---------------------------------------------------------------------------
def make_split(all_facts: list[dict]) -> dict:
    """index_eval is NEVER trained on. Day-5 LoRA sees `train` only."""
    rng = random.Random(VOCAB["split"]["seed"])
    ids = sorted(f["image_id"] for f in all_facts)
    rng.shuffle(ids)
    k = int(len(ids) * VOCAB["split"]["index_eval_frac"])
    return {"index_eval": sorted(ids[:k]), "train": sorted(ids[k:]),
            "note": "LoRA fine-tuning may ONLY use `train`. Reporting a number "
                    "computed on images the encoder was tuned on is data leakage."}


# ---------------------------------------------------------------------------
def load_facts() -> list[dict]:
    out = []
    mf = ROOT / "data" / "manifest.jsonl"
    if mf.exists():
        seen = set()
        for line in mf.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["image_id"] in seen:
                continue
            seen.add(r["image_id"]); out.append(facts_from_web(r))
    fp = ROOT / "data" / "fashionpedia.jsonl"
    if fp.exists():
        out += [facts_from_fashionpedia(json.loads(l)) for l in fp.read_text().splitlines() if l.strip()]
    return out


def main(facts: list[dict] | None = None) -> None:
    facts = facts or load_facts()
    if not facts:
        raise SystemExit("no facts — run data_collection/collect.py first (or --selftest)")

    labels = {}
    for q in QUERIES:
        judged = {f["image_id"]: relevance(q["predicate"], f) for f in facts}
        labels[q["id"]] = {i: r for i, r in judged.items() if r > 0}

    pairs = build_swap_pairs(facts)
    split = make_split(facts)

    (ROOT / "eval" / "labels.json").write_text(json.dumps(labels, indent=1))
    (ROOT / "eval" / "swap_pairs.json").write_text(json.dumps(pairs, indent=1))
    (ROOT / "data" / "split.json").write_text(json.dumps(split, indent=1))

    print(f"images        : {len(facts)}  (strong={sum(f['label_conf']=='strong' for f in facts)})")
    print(f"split         : index_eval={len(split['index_eval'])}  train={len(split['train'])}")
    print(f"swap pairs    : {len(pairs)}  <- binding-accuracy pairs")
    thin = [q['id'] for q in QUERIES if len(labels[q['id']]) < 3]
    for q in QUERIES:
        n2 = sum(v == 2 for v in labels[q["id"]].values())
        print(f"  {q['id']:9s} rel2={n2:3d} rel1={len(labels[q['id']])-n2:3d}  {q['text'][:44]}")
    if thin:
        print(f"\n!! too few positives for {thin} — add grid cells & re-collect")


# ---------------------------------------------------------------------------
def selftest() -> None:
    F = [
        {"image_id": "a1", "source": "web", "label_conf": "weak", "scene": "office", "formality": "formal",
         "garments": [{"type": "tie", "color": "red"}, {"type": "shirt", "color": "white"}]},   # Q5 correct
        {"image_id": "b1", "source": "web", "label_conf": "weak", "scene": "office", "formality": "formal",
         "garments": [{"type": "tie", "color": "white"}, {"type": "shirt", "color": "red"}]},   # Q5 SWAPPED
        {"image_id": "c1", "source": "web", "label_conf": "weak", "scene": "street", "formality": "casual",
         "garments": [{"type": "raincoat", "color": "yellow"}]},                                # Q1
        {"image_id": "d1", "source": "web", "label_conf": "weak", "scene": "park", "formality": "casual",
         "garments": [{"type": "shirt", "color": "blue"}]},                                     # Q3
        {"image_id": "e1", "source": "web", "label_conf": "weak", "scene": "office", "formality": "formal",
         "garments": [{"type": "shirt", "color": "red"}]},                                      # distractor
    ]
    P = {q["id"]: q["predicate"] for q in QUERIES}
    assert relevance(P["Q5"], F[0]) == 2, "correct binding must be rel=2"
    assert relevance(P["Q5"], F[1]) == 0, "SWAPPED binding must be rel=0 <-- the whole point"
    assert relevance(P["Q5"], F[4]) < 2, "single red shirt must not satisfy tie+shirt (one-to-one)"
    assert relevance(P["Q1"], F[2]) == 2 and relevance(P["Q3"], F[3]) == 2
    assert relevance(P["Q2"], F[0]) == 2, "shirt in formal office -> business attire"
    pairs = build_swap_pairs(F)
    assert any(p["query_id"] == "Q5" and p["positive_ids"] == ["a1"]
               and p["swapped_negative_ids"] == ["b1"] for p in pairs), pairs
    print("selftest OK — one-to-one binding enforced, swap pair (a1 vs b1) built")
    main(F)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    selftest() if a.selftest else main()

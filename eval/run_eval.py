"""Evaluation metrics for every system.

Metrics are chosen per query type. Narrow attribute queries have small relevant
sets and use Recall@5/@10. Broad scene/style queries match a large share of the
corpus, where recall is uninformative, so they use Precision@5/@10. Compositional
queries use binding accuracy: given a correct image A and its colour-swapped twin
B, score 1 iff A ranks above B (chance = 0.5), which isolates binding from every
other cue.

    python eval/run_eval.py --rankings artifacts/rankings_clip_global.json
    python eval/run_eval.py --selftest
"""

from __future__ import annotations

import argparse, json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NARROW = {"attribute", "complex_semantic"}
BROAD = {"contextual", "style_inference"}
COMP = {"compositional"}


def recall_at_k(ranked, rel, k):
    if not rel:
        return None
    return len(set(ranked[:k]) & rel) / min(len(rel), k)


def precision_at_k(ranked, rel, k):
    if not ranked:
        return 0.0
    return len(set(ranked[:k]) & rel) / k


def average_precision(ranked, rel):
    if not rel:
        return None
    hits, s = 0, 0.0
    for i, d in enumerate(ranked, 1):
        if d in rel:
            hits += 1
            s += hits / i
    return s / min(len(rel), len(ranked)) if hits else 0.0


def binding_accuracy(ranked, pos_ids, neg_ids):
    """Pairwise: every correct-binding image must outrank every swapped one."""
    pos = {d: i for i, d in enumerate(ranked)}
    miss = len(ranked) + 1
    wins = tot = 0
    for a in pos_ids:
        for b in neg_ids:
            ra, rb = pos.get(a, miss), pos.get(b, miss)
            if ra == rb == miss:
                continue                       # neither retrieved: no signal
            wins += ra < rb
            tot += 1
    return (wins / tot, tot) if tot else (None, 0)


def evaluate(rankings: dict, labels: dict, pairs: list, queries: list) -> dict:
    by_type: dict[str, list] = {}
    rows = []
    for q in queries:
        if q.get("swap_of"):                   # swap twins are probes, not queries
            continue
        qid, t = q["id"], q["type"]
        ranked = rankings.get(qid, [])
        rel = {i for i, r in labels.get(qid, {}).items() if r == 2}   # strict: rel=2 only
        row = {"id": qid, "type": t, "n_rel": len(rel), "text": q["text"]}

        if t in NARROW:
            row["R@5"] = recall_at_k(ranked, rel, 5)
            row["R@10"] = recall_at_k(ranked, rel, 10)
            row["_key"] = row["R@5"]
        elif t in BROAD:
            row["P@5"] = precision_at_k(ranked, rel, 5)
            row["P@10"] = precision_at_k(ranked, rel, 10)
            row["_key"] = row["P@5"]
        if t in COMP:
            row["P@5"] = precision_at_k(ranked, rel, 5)
            p = next((x for x in pairs if x["query_id"] == qid), None)
            if p:
                acc, n = binding_accuracy(ranked, p["positive_ids"], p["swapped_negative_ids"])
                row["binding"], row["n_cmp"] = acc, n
                row["difficulty"] = p.get("difficulty", "unspecified")
            row["_key"] = row.get("binding")
        row["AP"] = average_precision(ranked, rel)
        rows.append(row)
        by_type.setdefault(t, []).append(row)

    def mean(vals):
        v = [x for x in vals if x is not None]
        return sum(v) / len(v) if v else None

    def pooled_binding(rows_subset):
        """Comparison-weighted: a pair with n=48 counts more than one with n=16.
        Naive mean-of-rates would let a tiny pair swing the headline number."""
        w = sum((r["binding"] * r["n_cmp"]) for r in rows_subset if r.get("binding") is not None)
        n = sum(r["n_cmp"] for r in rows_subset if r.get("binding") is not None)
        return (w / n, n) if n else (None, 0)

    comp = [r for r in rows if r["type"] in COMP and r.get("binding") is not None]
    cc = [r for r in comp if r.get("difficulty") == "colour_critical"]
    ts = [r for r in comp if r.get("difficulty") == "type_separable"]
    bind_all, n_all = pooled_binding(comp)
    bind_cc, n_cc = pooled_binding(cc)
    bind_ts, n_ts = pooled_binding(ts)

    summary = {
        "narrow_R@5":  mean([r.get("R@5") for r in rows if r["type"] in NARROW]),
        "narrow_R@10": mean([r.get("R@10") for r in rows if r["type"] in NARROW]),
        "broad_P@5":   mean([r.get("P@5") for r in rows if r["type"] in BROAD]),
        "broad_P@10":  mean([r.get("P@10") for r in rows if r["type"] in BROAD]),
        "BINDING_ACC": bind_all, "binding_n": n_all,
        "BINDING_colour_critical": bind_cc, "binding_cc_n": n_cc,   # colour is only cue
        "BINDING_type_separable":  bind_ts, "binding_ts_n": n_ts,   # type/position leaks
        "mAP":         mean([r.get("AP") for r in rows]),
    }
    return {"summary": summary, "rows": rows}


def show(res: dict, name: str) -> None:
    s = res["summary"]
    f = lambda v: "  --  " if v is None else f"{v:.3f}"
    print(f"\n===== {name} =====")
    print(f"  narrow  R@5 {f(s['narrow_R@5'])}   R@10 {f(s['narrow_R@10'])}")
    print(f"  broad   P@5 {f(s['broad_P@5'])}   P@10 {f(s['broad_P@10'])}")
    print(f"  mAP         {f(s['mAP'])}")
    print(f"  BINDING ACC {f(s['BINDING_ACC'])}  (n={s.get('binding_n',0)})   <-- chance = 0.500")
    print(f"    ├─ colour-critical {f(s['BINDING_colour_critical'])}  (n={s.get('binding_cc_n',0)})   <-- colour is the ONLY cue")
    print(f"    └─ type-separable  {f(s['BINDING_type_separable'])}  (n={s.get('binding_ts_n',0)})   <-- type/position can leak")
    print("\n  per-query")
    for r in res["rows"]:
        m = (f"R@5 {f(r.get('R@5'))}" if r["type"] in NARROW else
             f"P@5 {f(r.get('P@5'))}")
        b = f"  bind {f(r.get('binding'))} (n={r.get('n_cmp',0)},{r.get('difficulty','?')[:4]})" if r["type"] in COMP else ""
        star = " *" if r.get("official") else ""
        print(f"   {r['id']:9s} {r['type']:17s} rel={r['n_rel']:3d}  {m}{b}  {r['text'][:34]}{star}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rankings", required=True)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    if not Path(a.rankings).exists():
        raise SystemExit(f"Rankings file {a.rankings} not found. Run a retriever first (see README).")
    R = json.loads(Path(a.rankings).read_text())
    L = json.loads((ROOT / "eval" / "labels.json").read_text())
    P = json.loads((ROOT / "eval" / "swap_pairs.json").read_text())
    Q = json.loads((ROOT / "eval" / "queries.json").read_text())["queries"]
    res = evaluate(R, L, P, Q)
    show(res, a.name or Path(a.rankings).stem)
    out = ROOT / "artifacts" / f"eval_{Path(a.rankings).stem}.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"\n-> {out.name}")


def selftest() -> None:
    """Prove the metric behaves: random ~0.5 binding, oracle = 1.0."""
    imgs = [f"i{n}" for n in range(200)]
    q = [{"id": "T1", "type": "compositional", "text": "red tie white shirt",
          "predicate": {}}]
    labels = {"T1": {i: 2 for i in imgs[:8]}}
    pairs = [{"query_id": "T1", "positive_ids": imgs[:8], "swapped_negative_ids": imgs[8:16]}]

    rng = random.Random(0)
    accs = []
    for _ in range(200):
        r = imgs[:]; rng.shuffle(r)
        accs.append(evaluate({"T1": r}, labels, pairs, q)["summary"]["BINDING_ACC"])
    mu = sum(accs) / len(accs)
    assert 0.45 < mu < 0.55, f"random binding acc should be ~0.5, got {mu}"

    oracle = imgs[:8] + imgs[8:]                       # all positives on top
    assert evaluate({"T1": oracle}, labels, pairs, q)["summary"]["BINDING_ACC"] == 1.0
    worst = imgs[8:16] + imgs[:8] + imgs[16:]          # all swaps on top
    assert evaluate({"T1": worst}, labels, pairs, q)["summary"]["BINDING_ACC"] == 0.0
    print(f"selftest OK — random={mu:.3f} (chance .500), oracle=1.000, worst=0.000")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--selftest", action="store_true")
    known, _ = ap.parse_known_args()
    selftest() if known.selftest else main()

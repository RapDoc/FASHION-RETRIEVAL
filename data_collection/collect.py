"""Collect fashion images on a stratified grid and record weak labels.

Each grid cell is a (environment, garment, colour, formality) tuple that becomes
a search query; the cell is stored with every returned image as its weak label,
which eval/labels.py later turns into relevance judgements. Resumable and
rate-limited so it can run unattended.

    python data_collection/collect.py --provider pexels --per-cell 8
    python data_collection/collect.py --plan-only   # inspect the grid, no network
"""

from __future__ import annotations

import argparse, hashlib, json, os, random, sys, time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
VOCAB = yaml.safe_load((ROOT / "configs" / "vocab.yaml").read_text())
OUT_IMG = ROOT / "data" / "images"
MANIFEST = ROOT / "data" / "manifest.jsonl"

RATE = {"pexels": 200, "unsplash": 50}  # requests/hour


# ----------------------------------------------------------------------
# 1. The GRID - stratified, not exhaustive.
# ----------------------------------------------------------------------
def build_grid() -> list[dict]:
    """
    Full cross-product is 6 env x 17 garment x 13 color x 4 formality = 5,304
    cells. We do not need it. We need (a) marginal coverage of every value on
    every axis, and (b) dense coverage of the regions the 5 eval queries live in.

    Returns a list of cells; each cell -> one search query -> ~N images.
    """
    G, C, E, F = (VOCAB[k] for k in ("garments", "colors", "environments", "formality"))
    cells: list[dict] = []

    def cell(env, garment, color, formality, tag, extra=""):
        q = f"person wearing {color} {garment} {extra} in {env.replace('_', ' ')}".strip()
        cells.append(dict(query=q, environment=env, garment=garment,
                          color=color, formality=formality, tag=tag))

    # -- (a) EVAL-Critical cells: over-sample exactly what we are graded on ----
    for c in ["yellow", "red", "blue", "green"]:                       # Q1
        cell("street", "raincoat", c, "casual", "eval_q1")
        cell("park", "raincoat", c, "casual", "eval_q1")
    for c in ["navy", "gray", "black", "white", "blue"]:               # Q2
        cell("office", "blazer", c, "formal", "eval_q2")
        cell("office", "suit", c, "formal", "eval_q2")
    for c in ["blue", "white", "red", "green"]:                        # Q3
        cell("park", "shirt", c, "casual", "eval_q3", extra="sitting on a bench")
    for c in ["blue", "gray", "black", "beige"]:                       # Q4
        cell("street", "hoodie", c, "casual", "eval_q4")
        cell("street", "jeans", c, "casual", "eval_q4")
        cell("street", "t-shirt", c, "casual", "eval_q4")
    # Two-garment queries: collect both colour bindings so binding accuracy has a
    # correct image and its swapped counterpart to compare.
    PAIRS = [                                # (g1, c1, g2, c2, env, formality)
        ("tie",    "red",   "shirt",   "white", "office", "formal"),          # Q5
        ("blazer", "navy",  "shirt",   "white", "office", "formal"),          # C01
        ("hoodie", "red",   "jeans",   "black", "street", "casual"),          # C02
        ("shirt",  "blue",  "pants",   "beige", "street", "business_casual"), # C03
        ("jacket", "green", "t-shirt", "white", "street", "casual"),          # C04
        ("coat",   "black", "sweater", "yellow", "street", "casual"),         # C05
    ]
    for g1, c1, g2, c2, env, form in PAIRS:
        for a, b in [(c1, c2), (c2, c1)]:    # <- correct binding AND its swap
            cells.append(dict(
                query=f"person wearing a {a} {g1} and {b} {g2} in {env.replace('_',' ')}",
                environment=env, garment=g1, color=a, formality=form,
                tag="compositional_pair",
                second_garment={"type": g2, "color": b},
            ))
    # extra Q5 tie/shirt bindings for a denser hard-negative pool
    for tie_c, shirt_c in [("navy", "white"), ("white", "navy"),
                           ("black", "white"), ("red", "blue"), ("blue", "red")]:
        cells.append(dict(
            query=f"man in a {tie_c} tie and a {shirt_c} shirt formal office",
            environment="office", garment="tie", color=tie_c, formality="formal",
            tag="eval_q5_swap",
            second_garment={"type": "shirt", "color": shirt_c},
        ))

    # -- ATTRIBUTE queries (A01-A06): random coverage sampling does not reliably
    # hit these, and a query with <3 positives is a meaningless eval row.
    for g, c, env, form in [("dress", "red", "formal_event", "formal"),        # A01
                            ("sweater", "yellow", "cafe", "casual"),           # A02
                            ("jacket", "blue", "street", "casual"),            # A03 denim
                            ("jacket", "black", "street", "casual"),           # A04
                            ("shirt", "black", "street", "casual"),            # A04
                            ("scarf", "pink", "street", "casual"),             # A05
                            ("suit", "gray", "office", "formal")]:             # A06
        for _ in range(2):
            cell(env, g, c, form, "eval_attribute")
    cells.append(dict(query="person in an all black outfit black jacket black jeans street",
                      environment="street", garment="jacket", color="black", formality="casual",
                      tag="eval_attribute", second_garment={"type": "jeans", "color": "black"}))

    # -- (b) MARGINAL coverage: every axis value appears with varied partners ---
    rng = random.Random(VOCAB["split"]["seed"])
    for g in G:
        for c in rng.sample(list(C), 3):
            env = rng.choice(list(E))
            cell(env, g, c, G[g]["formality_prior"], "coverage_garment")
    for c in C:                       # every colour, several garments
        for g in rng.sample(list(G), 2):
            cell(rng.choice(list(E)), g, c, G[g]["formality_prior"], "coverage_color")
    for env in E:                     # every environment
        for _ in range(4):
            g = rng.choice(list(G))
            cell(env, g, rng.choice(list(C)), G[g]["formality_prior"], "coverage_env")
    for f in F:                       # every formality (the axis we nearly forgot)
        for _ in range(6):
            g = rng.choice([k for k, v in G.items() if v["formality_prior"] == f] or list(G))
            cells.append(dict(query=f"{f.replace('_',' ')} outfit {g} {rng.choice(list(E)).replace('_',' ')}",
                              environment=rng.choice(list(E)), garment=g,
                              color=None, formality=f, tag="coverage_formality"))

    # dedupe on query text
    seen, uniq = set(), []
    for c in cells:
        if c["query"] not in seen:
            seen.add(c["query"]); uniq.append(c)
    return uniq


# ----------------------------------------------------------------------
# 2. PROVIDERS
# ----------------------------------------------------------------------
def search_pexels(q: str, n: int, key: str) -> list[dict]:
    r = requests.get("https://api.pexels.com/v1/search",
                     headers={"Authorization": key},
                     params={"query": q, "per_page": n, "orientation": "portrait"},
                     timeout=20)
    r.raise_for_status()
    return [{"url": p["src"]["large"], "ext_id": f"pexels_{p['id']}",
             "credit": p["photographer"]} for p in r.json().get("photos", [])]


def search_unsplash(q: str, n: int, key: str) -> list[dict]:
    r = requests.get("https://api.unsplash.com/search/photos",
                     params={"query": q, "per_page": n, "orientation": "portrait"},
                     headers={"Authorization": f"Client-ID {key}"}, timeout=20)
    r.raise_for_status()
    return [{"url": p["urls"]["regular"], "ext_id": f"unsplash_{p['id']}",
             "credit": p["user"]["name"]} for p in r.json().get("results", [])]


PROVIDERS = {"pexels": search_pexels, "unsplash": search_unsplash}


# ----------------------------------------------------------------------
# 3. RUN - resumable, rate-limited
# ----------------------------------------------------------------------
def load_done() -> set[str]:
    if not MANIFEST.exists():
        return set()
    return {json.loads(l)["cell_query"] for l in MANIFEST.read_text().splitlines() if l.strip()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=PROVIDERS, default="pexels")
    ap.add_argument("--per-cell", type=int, default=8)
    ap.add_argument("--plan-only", action="store_true")
    a = ap.parse_args()

    grid = build_grid()
    print(f"grid: {len(grid)} cells x {a.per_cell} = ~{len(grid)*a.per_cell} images target")
    if a.plan_only:
        from collections import Counter
        print("  by tag:", dict(Counter(c["tag"] for c in grid)))
        for c in grid[:6]:
            print("   e.g.", c["query"])
        return

    key = os.environ.get(f"{a.provider.upper()}_API_KEY")
    if not key:
        sys.exit(f"set {a.provider.upper()}_API_KEY")

    OUT_IMG.mkdir(parents=True, exist_ok=True)
    done, sleep_s = load_done(), 3600.0 / RATE[a.provider]
    todo = [c for c in grid if c["query"] not in done]
    print(f"resuming: {len(done)} cells done, {len(todo)} to go "
          f"(~{len(todo)*sleep_s/3600:.1f}h at {RATE[a.provider]}/hr)")

    with MANIFEST.open("a") as mf:
        for i, cell in enumerate(todo):
            try:
                hits = PROVIDERS[a.provider](cell["query"], a.per_cell, key)
            except Exception as e:                       # rate-limit / transient
                print(f"  ! {cell['query'][:40]}: {e} — backing off 60s")
                time.sleep(60); continue

            for h in hits:
                iid = hashlib.md5(h["ext_id"].encode()).hexdigest()[:12]
                p = OUT_IMG / f"{iid}.jpg"
                if not p.exists():
                    try:
                        p.write_bytes(requests.get(h["url"], timeout=30).content)
                    except Exception:
                        continue
                # ---- The WEAK LABEL: the grid cell that produced this image ----
                rec = {"image_id": iid, "path": str(p.relative_to(ROOT)).replace("\\", "/"),
                       "source": a.provider, "credit": h["credit"],
                       "cell_query": cell["query"],
                       "weak": {k: cell.get(k) for k in
                                ("environment", "garment", "color", "formality",
                                 "second_garment", "tag")}}
                mf.write(json.dumps(rec) + "\n")
            mf.flush()
            print(f"[{i+1}/{len(todo)}] {len(hits):2d} imgs  {cell['query'][:52]}")
            time.sleep(sleep_s)


if __name__ == "__main__":
    main()

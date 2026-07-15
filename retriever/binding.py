"""Hungarian binding reranker: the core of the compositional retrieval.

A global image vector treats "red tie + white shirt" and "white tie + red shirt"
as the same bag of attributes. Here we instead assign each queried garment to one
image region (one-to-one), so a colour-swapped image cannot satisfy both clauses
and is forced to rank lower.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
VOCAB = yaml.safe_load((ROOT / "configs" / "vocab.yaml").read_text())

# Which query garment types each SegFormer class can plausibly be.
SEG_FAMILY = {"upper-clothes": {"shirt", "t-shirt", "blazer", "suit", "hoodie",
                                "sweater", "coat", "raincoat", "jacket", "tie"},
              "pants": {"pants", "jeans", "shorts"}, "skirt": {"skirt"},
              "dress": {"dress"}, "scarf": {"scarf"}, "hat": {"hat"}}

# Scoring weights (mirrored in configs/scoring.yaml for tuning).
ALPHA, BETA, GAMMA = 0.55, 0.30, 0.15
LAB_W = np.array([0.55, 1.0, 1.0], "float32")   # down-weight lightness vs hue/chroma
LAB_SCALE = 40.0                                 # colour distance at which proximity -> 0


def lab_prox(lab_r, lab_q) -> float:
    """Perceptual colour closeness in [0,1] from two CIELAB triples."""
    if lab_r is None or lab_q is None:
        return 0.0
    d = np.sqrt((((np.array(lab_r) - np.array(lab_q)) * LAB_W) ** 2).sum())
    return float(max(0.0, 1.0 - d / LAB_SCALE))


def type_match(qtype: str, seg_class: str) -> float:
    """1 if the query garment type is compatible with the region's SegFormer class."""
    return 1.0 if qtype in SEG_FAMILY.get(seg_class, set()) else 0.0


def binding_score(query_garments: list[dict], regions: list[dict],
                  region_emb: np.ndarray, garment_text_emb: dict,
                  a=ALPHA, b=BETA, g=GAMMA) -> tuple[float, float]:
    """Score how well an image's regions bind to the queried garments.

    Returns (score, coverage). Coverage is the fraction of queried garments that
    found a region; the caller uses it to fall back to global recall when
    segmentation could not isolate all garments (e.g. a coat merged with a sweater).
    """
    m, n = len(query_garments), len(regions)
    if m == 0 or n == 0:
        return 0.0, 0.0

    # Cost matrix: query garment i vs image region j.
    S = np.zeros((m, n), "float32")
    for i, q in enumerate(query_garments):
        key = f"a {q.get('color') or ''} {q['type']}".replace("  ", " ").strip()
        tvec = garment_text_emb.get(key)
        for j, r in enumerate(regions):
            emb_sim = float(region_emb[r["emb_idx"]] @ tvec) if tvec is not None else 0.0
            S[i, j] = (a * emb_sim
                       + b * lab_prox(r.get("color_lab"), q.get("color_lab"))
                       + g * type_match(q["type"], r["seg_class"]))

    # Optimal one-to-one assignment maximising total pair score.
    rows, cols = linear_sum_assignment(-S)
    matched = float(S[rows, cols].sum())
    coverage = len(rows) / m
    score = matched / max(len(rows), 1)
    return score, coverage


if __name__ == "__main__":
    # Sanity checks. A real region embedding encodes colour and garment type
    # together, so text("a red tie") and text("a white tie") point in different
    # directions; the discriminating signal is that the tie region carries the
    # queried colour. Dims below stand for [red, white, tie-ness, shirt-ness].
    def unit(v):
        v = np.array(v, "float32")
        return v / (np.linalg.norm(v) + 1e-9)

    txt = {"a red tie": unit([1, 0, 1, 0]), "a white shirt": unit([0, 1, 0, 1])}
    emb = np.stack([unit([1, 0, 1, 0]),   # red tie
                    unit([0, 1, 0, 1]),   # white shirt
                    unit([0, 1, 1, 0]),   # white tie (swapped image)
                    unit([1, 0, 0, 1])])  # red shirt (swapped image)
    q = [{"type": "tie", "color": "red", "color_lab": [53, 80, 67]},
         {"type": "shirt", "color": "white", "color_lab": [100, 0, 0]}]
    correct = [{"seg_class": "upper-clothes", "color_lab": [53, 80, 67], "emb_idx": 0},
               {"seg_class": "upper-clothes", "color_lab": [100, 0, 0], "emb_idx": 1}]
    swapped = [{"seg_class": "upper-clothes", "color_lab": [100, 0, 0], "emb_idx": 2},
               {"seg_class": "upper-clothes", "color_lab": [53, 80, 67], "emb_idx": 3}]

    sc, _ = binding_score(q, correct, emb, txt)
    ss, _ = binding_score(q, swapped, emb, txt)
    assert sc > ss, "correct binding must outrank swapped"
    print(f"correct={sc:.3f}  swapped={ss:.3f}  ->  correct wins by {sc - ss:.3f}")

    txt_b = {"a red tie": unit([0, 0, 1, 0]), "a white shirt": unit([0, 0, 0, 1])}
    emb_b = np.stack([unit([0, 0, 1, 0]), unit([0, 0, 0, 1]),
                      unit([0, 0, 1, 0]), unit([0, 0, 0, 1])])
    sc2, _ = binding_score(q, correct, emb_b, txt_b)
    ss2, _ = binding_score(q, swapped, emb_b, txt_b)
    assert sc2 > ss2, "LAB term must separate swaps when the embedding cannot"
    print(f"colour-blind encoder: correct={sc2:.3f}  swapped={ss2:.3f}  (LAB tiebreak holds)")

    _, cov = binding_score(q, correct[:1], emb, txt)
    assert abs(cov - 0.5) < 1e-6
    print(f"two garments vs one region -> coverage {cov:.2f} (caller defers to global)")

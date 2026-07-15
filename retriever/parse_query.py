"""Parse a natural-language query into structured intent.

Produces the same schema the image side uses: a list of garments (each with a
type, colour, and colour LAB), plus scene and formality. A regex/keyword parser
over the vocab is the default and needs no API key; if GROQ_API_KEY is set an LLM
parser is used instead, falling back to the regex parser on any error.
"""

from __future__ import annotations

import json, os, re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
VOCAB = yaml.safe_load((ROOT / "configs" / "vocab.yaml").read_text())

COLORS = VOCAB["colors"]
FAMILIES = VOCAB.get("color_families", {})
GARMENTS = VOCAB["garments"]
ENVS = VOCAB["environments"]
FORMS = VOCAB["formality"]


def _alias_lookup(section: dict) -> dict[str, str]:
    m = {}
    for canon, meta in section.items():
        m[canon.replace("_", " ")] = canon
        for a in (meta or {}).get("aliases", []) or []:
            m[a.lower()] = canon
    return m

COLOR_A = _alias_lookup(COLORS)
GARMENT_A = _alias_lookup(GARMENTS)
ENV_A = _alias_lookup(ENVS)
FORM_A = _alias_lookup(FORMS)


def family(color: str | None) -> str | None:
    return FAMILIES.get(color, color) if color else None


def color_lab(color: str | None):
    return COLORS[color]["lab"] if color and color in COLORS else None


def _find_all(text: str, table: dict) -> list[tuple[int, str]]:
    """Return (position, canonical) for every alias found, longest-match first."""
    hits = []
    for phrase in sorted(table, key=len, reverse=True):
        for m in re.finditer(rf"\b{re.escape(phrase)}\b", text):
            hits.append((m.start(), table[phrase]))
    return hits


def parse_fallback(text: str) -> dict:
    t = text.lower()

    # garments in order of appearance
    g_hits = sorted(_find_all(t, GARMENT_A))
    c_hits = sorted(_find_all(t, COLOR_A))

    # bind each garment to the NEAREST-PRECEDING colour word (English puts the
    # adjective before the noun: "red tie", "white shirt"). If no colour precedes
    # a garment, it is left uncoloured rather than mis-bound.
    garments = []
    used_c = set()
    for pos, gtype in g_hits:
        best = None
        for i, (cpos, cname) in enumerate(c_hits):
            if i in used_c:
                continue
            if cpos < pos:                       # colour appears before garment
                if best is None or cpos > c_hits[best][0]:
                    best = i
        color = None
        if best is not None and pos - c_hits[best][0] < 25:   # within ~4 words
            color = c_hits[best][1]; used_c.add(best)
        garments.append({"type": gtype, "color": color,
                         "color_family": family(color), "color_lab": color_lab(color)})

    scene = next((c for _, c in sorted(_find_all(t, ENV_A))), None)
    formality = next((c for _, c in sorted(_find_all(t, FORM_A))), None)
    # a garment's formality prior can supply scene-less formality (e.g. "suit")
    if not formality and garments:
        priors = [GARMENTS[g["type"]]["formality_prior"] for g in garments if g["type"] in GARMENTS]
        if priors:
            formality = max(set(priors), key=priors.count)

    return {"garments": garments, "scene": scene, "formality": formality, "raw": text}


def parse_groq(text: str) -> dict:
    """Optional LLM path. Falls back silently if unavailable."""
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        return parse_fallback(text)
    try:
        from groq import Groq
        schema = ('Return ONLY JSON: {"garments":[{"type","color"}],"scene","formality"}. '
                  f'garment types: {list(GARMENTS)}. colors: {list(COLORS)}. '
                  f'scenes: {list(ENVS)}. formality: {list(FORMS)}. '
                  'Bind each colour to its garment. Null if absent.')
        r = Groq(api_key=key).chat.completions.create(
            model="llama-3.3-70b-versatile", temperature=0,
            messages=[{"role": "system", "content": schema},
                      {"role": "user", "content": text}])
        raw = json.loads(re.sub(r"```json|```", "", r.choices[0].message.content).strip())
        out = {"garments": [], "scene": raw.get("scene"),
               "formality": raw.get("formality"), "raw": text}
        for g in raw.get("garments", []):
            c = g.get("color")
            out["garments"].append({"type": g.get("type"), "color": c,
                                    "color_family": family(c), "color_lab": color_lab(c)})
        return out
    except Exception:
        return parse_fallback(text)               # never let the LLM break the demo


def parse(text: str, use_groq: bool = False) -> dict:
    return parse_groq(text) if use_groq else parse_fallback(text)


if __name__ == "__main__":
    for q in ["a red tie and a white shirt in a formal setting",
              "someone wearing a blue shirt sitting on a park bench",
              "a black coat over a yellow sweater",
              "professional business attire inside a modern office",
              "casual weekend outfit for a city walk",
              "a person in a bright yellow raincoat"]:
        p = parse(q)
        gs = ", ".join(f"{g['color'] or '?'}:{g['type']}" for g in p["garments"]) or "(none)"
        print(f"{q}\n   -> [{gs}]  scene={p['scene']} formality={p['formality']}\n")

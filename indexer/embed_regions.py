"""Region indexer: garment masks, region embeddings, and per-garment colour.

For each image: SegFormer-clothes produces a mask per garment class; each garment
is embedded from its masked crop with the same encoder as the global index; and a
dominant colour is extracted by k-means over the masked pixels in CIELAB. The
masked colour gives the reranker a per-garment colour signal the global vector
lacks.

    python indexer/embed_regions.py --encoder fashionsiglip
    python indexer/embed_regions.py --encoder fashionsiglip --debug 5   # save mask overlays
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
VOCAB = yaml.safe_load((ROOT / "configs" / "vocab.yaml").read_text())

# SegFormer-clothes (mattmdjaga/segformer_b2_clothes) label map.
SEG_ID = "mattmdjaga/segformer_b2_clothes"
SEG_LABELS = {0: "background", 1: "hat", 2: "hair", 3: "sunglasses",
              4: "upper-clothes", 5: "skirt", 6: "pants", 7: "dress",
              8: "belt", 9: "left-shoe", 10: "right-shoe", 11: "face",
              12: "left-leg", 13: "right-leg", 14: "left-arm", 15: "right-arm",
              16: "bag", 17: "scarf"}
# SegFormer class -> our garment family. SegFormer-clothes has no separate coat
# class, so coats/jackets/blazers all segment as `upper-clothes`; a coat over a
# sweater therefore returns a single merged upper mask rather than two regions.
SEG_TO_GARMENT = {"upper-clothes": "upper", "dress": "dress",
                  "pants": "pants", "skirt": "skirt", "scarf": "scarf", "hat": "hat"}
GARMENT_CLASSES = {4, 5, 6, 7, 17, 1}          # upper, skirt, pants, dress, scarf, hat
MIN_REGION_FRAC = 0.01                          # ignore garments <1% of image


def load_segmenter(device):
    from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
    proc = SegformerImageProcessor.from_pretrained(SEG_ID)
    model = AutoModelForSemanticSegmentation.from_pretrained(SEG_ID).to(device).eval()

    def segment(img: Image.Image) -> np.ndarray:
        x = proc(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**x).logits
        up = torch.nn.functional.interpolate(
            logits, size=img.size[::-1], mode="bilinear", align_corners=False)
        return up.argmax(1)[0].cpu().numpy()      # H x W of class ids

    return segment


# ---- masked dominant colour ------------------------------------------------
COLOR_LAB = {c: np.array(m["lab"], dtype="float32") for c, m in VOCAB["colors"].items()}


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """rgb uint8 [...,3] -> CIELAB. Minimal sRGB->XYZ->Lab, D65."""
    a = rgb.astype("float32") / 255.0
    m = a > 0.04045
    a = np.where(m, ((a + 0.055) / 1.055) ** 2.4, a / 12.92)
    X = a @ np.array([0.4124, 0.3576, 0.1805], "float32")
    Y = a @ np.array([0.2126, 0.7152, 0.0722], "float32")
    Z = a @ np.array([0.0193, 0.1192, 0.9505], "float32")
    xyz = np.stack([X / 0.95047, Y, Z / 1.08883], -1)
    m2 = xyz > 0.008856
    f = np.where(m2, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    L = 116 * f[..., 1] - 16
    A = 500 * (f[..., 0] - f[..., 1])
    B = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, A, B], -1)


def dominant_color(img_rgb: np.ndarray, mask: np.ndarray, k: int = 4):
    """k-means in LAB over MASKED pixels; return nearest vocab colour.

    Two robustness fixes learned from debug images:
      * SHADOW REJECTION: garment folds photograph as near-black low-chroma
        pixels; the largest cluster is often shadow, not the true colour. We
        drop clusters with L<18 AND chroma<12 before voting, unless the garment
        really is black (all clusters dark -> keep them).
      * We vote by cluster SIZE among the surviving (non-shadow) clusters.
    """
    px = img_rgb[mask]
    if len(px) < 30:
        return None, None, 0.0
    lab = _rgb_to_lab(px.reshape(-1, 3)).reshape(-1, 3)
    rng = np.random.default_rng(0)
    cen = lab[rng.choice(len(lab), min(k, len(lab)), replace=False)]
    for _ in range(12):
        d = ((lab[:, None] - cen[None]) ** 2).sum(-1)
        a = d.argmin(1)
        newc = np.array([lab[a == j].mean(0) if (a == j).any() else cen[j]
                         for j in range(len(cen))])
        if np.allclose(newc, cen):
            break
        cen = newc
    counts = np.bincount(a, minlength=len(cen)).astype("float32")

    # shadow rejection
    L = cen[:, 0]
    chroma = np.sqrt(cen[:, 1] ** 2 + cen[:, 2] ** 2)
    is_shadow = (L < 18) & (chroma < 12)
    if not is_shadow.all():                       # keep shadow only if that's ALL there is
        counts = counts * (~is_shadow)

    if counts.sum() == 0:
        return "black", [0.0, 0.0, 0.0], 1.0      # genuinely dark garment
    j = int(counts.argmax())
    dom, frac = cen[j], float(counts[j] / counts.sum())

    # Snap with L down-weighted 0.55x: shadow/lighting mostly perturbs lightness,
    # garment identity lives in hue/chroma (a,b). Full-weight L made shadowed-red
    # snap to brown. Known residual: very dark navy can read as black (they are
    # perceptually close); no eval pair depends on that distinction.
    names = list(COLOR_LAB); C = np.stack([COLOR_LAB[n] for n in names])
    W = np.array([0.55, 1.0, 1.0], "float32")
    name = names[int((((C - dom) * W) ** 2).sum(1).argmin())]
    return name, dom.tolist(), frac


def load_encoder_img(name, device):
    import sys; sys.path.insert(0, str(ROOT))
    from indexer.embed_global import load_encoder
    enc_img, _ = load_encoder(name, device)
    return enc_img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="fashionsiglip")
    ap.add_argument("--debug", type=int, default=0)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not (ROOT / "data" / "manifest.jsonl").exists():
        raise SystemExit("No dataset found. See README - run data_collection/collect.py first.")

    split = json.loads((ROOT / "data" / "split.json").read_text())
    keep = set(split["index_eval"])
    recs = [json.loads(l) for l in (ROOT / "data" / "manifest.jsonl").read_text().splitlines() if l.strip()]
    seen, imgs = set(), []
    for r in recs:
        if r["image_id"] in keep and r["image_id"] not in seen:
            seen.add(r["image_id"]); imgs.append(r)
    print(f"regions for {len(imgs)} images | encoder={a.encoder} | device={device}")

    segment = load_segmenter(device)
    enc_img = load_encoder_img(a.encoder, device)

    region_recs, crops, crop_ptr = [], [], []
    dbg = 0
    for n, r in enumerate(imgs):
        try:
            img = Image.open(ROOT / r["path"]).convert("RGB")
        except Exception:
            continue
        arr = np.asarray(img)
        seg = segment(img)
        H, W = seg.shape
        regions = []
        for cid in GARMENT_CLASSES:
            mask = seg == cid
            frac = mask.mean()
            if frac < MIN_REGION_FRAC:
                continue
            ys, xs = np.where(mask)
            box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            # masked crop: zero out non-garment pixels so the region embedding
            # sees the garment, not its neighbours
            crop = arr.copy()
            crop[~mask] = 0
            crop = crop[box[1]:box[3] + 1, box[0]:box[2] + 1]
            if crop.size == 0:
                continue
            cname, clab, cfrac = dominant_color(arr, mask)
            regions.append({"seg_class": SEG_LABELS[cid], "box": box,
                            "area_frac": round(float(frac), 4),
                            "color_name": cname, "color_lab": clab,
                            "color_frac": round(cfrac, 3),
                            "emb_idx": len(crops)})
            crops.append(Image.fromarray(crop))
        region_recs.append({"image_id": r["image_id"], "regions": regions})

        if a.debug and dbg < a.debug and regions:
            (ART / "debug").mkdir(parents=True, exist_ok=True)
            over = arr.copy()
            for reg in regions:
                x0, y0, x1, y1 = reg["box"]
                over[y0:y1, x0] = over[y0:y1, x1] = [255, 0, 0]
                over[y0, x0:x1] = over[y1, x0:x1] = [255, 0, 0]
            Image.fromarray(over).save(ART / "debug" / f"{r['image_id']}.jpg")
            print("  dbg", r["image_id"], [(reg["seg_class"], reg["color_name"]) for reg in regions])
            dbg += 1
        if n % 50 == 0:
            print(f"  {n}/{len(imgs)}  crops={len(crops)}", end="\r")

    # embed all region crops in batches
    E = []
    for i in range(0, len(crops), 64):
        E.append(enc_img(crops[i:i + 64]))
    RE = np.concatenate(E).astype("float32") if E else np.zeros((0, 512), "float32")

    np.save(ART / f"region_emb_{a.encoder}.npy", RE)
    (ART / f"regions_{a.encoder}.jsonl").write_text(
        "\n".join(json.dumps(x) for x in region_recs))
    n_reg = sum(len(x["regions"]) for x in region_recs)
    print(f"\n{len(region_recs)} images, {n_reg} regions "
          f"({n_reg/max(len(region_recs),1):.1f}/img), emb={RE.shape}")


if __name__ == "__main__":
    main()

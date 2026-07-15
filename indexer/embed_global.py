"""Global-vector indexer: one embedding per image.

Encoder is swappable via --encoder (clip is the baseline, fashionsiglip the
fashion-tuned model); both write a .npy of image vectors plus an id index.

    python indexer/embed_global.py --encoder clip
    python indexer/embed_global.py --encoder fashionsiglip
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"

ENCODERS = {
    # name -> (hf_id, loader) 'open_clip' models need open_clip_torch installed
    "clip":           {"hf": "openai/clip-vit-base-patch32", "lib": "transformers"},
    "fashionsiglip":  {"hf": "Marqo/marqo-fashionSigLIP",    "lib": "open_clip"},
}


def load_encoder(name: str, device: str):
    spec = ENCODERS[name]
    if spec["lib"] == "transformers":
        from transformers import CLIPModel, CLIPProcessor
        m = CLIPModel.from_pretrained(spec["hf"]).to(device).eval()
        p = CLIPProcessor.from_pretrained(spec["hf"])

        def _t(v):
            # Some transformers versions return a ModelOutput rather than a tensor.
            if torch.is_tensor(v):
                return v
            for k in ("image_embeds", "text_embeds", "pooler_output"):
                if hasattr(v, k):
                    return getattr(v, k)
            raise TypeError(type(v))

        def enc_img(imgs):
            x = p(images=imgs, return_tensors="pt").to(device)
            with torch.no_grad():
                v = _t(m.get_image_features(pixel_values=x["pixel_values"]))
            return torch.nn.functional.normalize(v, dim=-1).cpu().numpy()

        def enc_txt(txts):
            x = p(text=txts, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                v = _t(m.get_text_features(input_ids=x["input_ids"],
                                           attention_mask=x["attention_mask"]))
            return torch.nn.functional.normalize(v, dim=-1).cpu().numpy()
    else:
        import open_clip
        m, _, pre = open_clip.create_model_and_transforms(f"hf-hub:{spec['hf']}")
        tok = open_clip.get_tokenizer(f"hf-hub:{spec['hf']}")
        m = m.to(device).eval()

        def enc_img(imgs):
            x = torch.stack([pre(i) for i in imgs]).to(device)
            with torch.no_grad():
                v = m.encode_image(x)
            return torch.nn.functional.normalize(v, dim=-1).cpu().numpy()

        def enc_txt(txts):
            with torch.no_grad():
                v = m.encode_text(tok(txts).to(device))
            return torch.nn.functional.normalize(v, dim=-1).cpu().numpy()

    return enc_img, enc_txt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=ENCODERS, default="clip")
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not (ROOT / "data" / "manifest.jsonl").exists():
        raise SystemExit("No dataset found. See README - run data_collection/collect.py first.")
    print(f"encoder={a.encoder}  device={device}")

    # Only index the index_eval split; the train split is held out for fine-tuning.
    split = json.loads((ROOT / "data" / "split.json").read_text())
    keep = set(split["index_eval"])
    recs = [json.loads(l) for l in (ROOT / "data" / "manifest.jsonl").read_text().splitlines() if l.strip()]
    seen, imgs = set(), []
    for r in recs:
        if r["image_id"] in keep and r["image_id"] not in seen:
            seen.add(r["image_id"]); imgs.append(r)
    print(f"indexing {len(imgs)} images (index_eval split; {len(split['train'])} held out)")

    enc_img, _ = load_encoder(a.encoder, device)
    vecs, ids, bad = [], [], 0
    for i in range(0, len(imgs), a.batch):
        chunk, pil = imgs[i:i + a.batch], []
        for r in chunk:
            try:
                pil.append(Image.open(ROOT / r["path"]).convert("RGB"))
            except Exception:
                bad += 1; pil.append(None)
        ok = [(r, p) for r, p in zip(chunk, pil) if p is not None]
        if not ok:
            continue
        vecs.append(enc_img([p for _, p in ok]))
        ids += [r["image_id"] for r, _ in ok]
        print(f"  {min(i + a.batch, len(imgs))}/{len(imgs)}", end="\r")

    V = np.concatenate(vecs).astype("float32")
    ART.mkdir(exist_ok=True)
    np.save(ART / f"global_{a.encoder}.npy", V)
    (ART / f"index_{a.encoder}.json").write_text(json.dumps({"image_ids": ids, "encoder": a.encoder}))
    print(f"\nsaved {V.shape} -> artifacts/global_{a.encoder}.npy   ({bad} unreadable)")


if __name__ == "__main__":
    main()

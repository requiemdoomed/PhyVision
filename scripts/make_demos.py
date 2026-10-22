"""Render the named demo people shown in the PhyVision gallery.

These are fictional, procedurally rendered people (not photos of real
individuals). Each gets a JPEG, a reference mask, and, once a trained model
exists, a pre-computed prediction stored in web/demo/demos.json so the gallery
also works in offline/static mode.

    python scripts/make_demos.py            # render images + run predictions
    python scripts/make_demos.py --no-predict
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402

from phyvision.config import CHECKPOINT, DEMO_DIR  # noqa: E402
from phyvision.synth import generate_person  # noqa: E402

FULL = dict(body_frac=0.84, framing=("full", 1.0), watermark=False, out_w=600, out_h=900, ss=2,
            pose="apart")  # ideal photo: arms slightly away from the body, as the photo guide asks
DEMOS = [
    # slug, display name, intended physique, sex, seed, render options
    ("john_bahadur", "John Bahadur", "athletic", "male", 501),
    ("aarav_shrestha", "Aarav Shrestha", "skinny", "male", 502),
    ("maya_gurung", "Maya Gurung", "normal", "female", 503),
    ("sofia_martinez", "Sofia Martinez", "athletic", "female", 504),
    ("liam_oconnor", "Liam O'Connor", "obese", "male", 505),
    ("priya_karki", "Priya Karki", "skinny", "female", 506),
    ("chen_wei", "Chen Wei", "normal", "male", 509),
    ("grace_thompson", "Grace Thompson", "obese", "female", 508),
    # supported non-ideal photos: cropped framing and a flex pose
    ("ethan_brooks", "Ethan Brooks", "athletic", "male", 646, dict(framing=("thigh", 0.64), pose="one_flex", out_w=760)),
    ("anika_rai", "Anika Rai", "normal", "female", 511, dict(framing=("knees", 0.78), pose="down")),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-predict", action="store_true")
    a = ap.parse_args()
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for slug, name, label, sex, seed, *opt in DEMOS:
        kw = dict(FULL)
        kw.update(opt[0] if opt else {})
        img, mask, meta = generate_person(label, seed, sex=sex, **kw)
        cv2.imwrite(str(DEMO_DIR / f"{slug}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(DEMO_DIR / f"{slug}_mask.png"), mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        out.append({"id": slug, "name": name, "image": f"demo/{slug}.jpg", "mask": f"demo/{slug}_mask.png",
                    "intended": label, "sex": sex, "framing": meta["framing"], "pose": str(meta["arm_pose"]),
                    "prediction": None})
        print("rendered", name)

    if not a.no_predict and CHECKPOINT.exists():
        from phyvision.inference import Predictor
        p = Predictor()
        for d in out:
            r = p.predict_file(DEMO_DIR / f"{d['id']}.jpg", DEMO_DIR / f"{d['id']}_mask.png")
            d["prediction"] = {k: r[k] for k in ("label", "label_display", "confidence", "probabilities",
                                                  "runner_up", "borderline", "metrics", "warnings", "framing", "checks")}
            d["prediction"]["segmentation"] = "reference"
            from phyvision.silhouette import normalise, overlay
            ov = overlay(r["_rgb"], r["_mask"])
            cv2.imwrite(str(DEMO_DIR / f"{d['id']}_overlay.jpg"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
            cv2.imwrite(str(DEMO_DIR / f"{d['id']}_silhouette.png"), (normalise(r["_mask"]) * 255).astype("uint8"))
            d["overlay"] = f"demo/{d['id']}_overlay.jpg"
            d["silhouette"] = f"demo/{d['id']}_silhouette.png"
            print(f"{d['name']:16s} -> {r['label']:9s} {r['confidence'] * 100:5.1f}%")
    (DEMO_DIR / "demos.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

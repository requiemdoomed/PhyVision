"""Classify one image from the command line.

    python scripts/predict.py path/to/photo.jpg [--mask mask.png] [--seg grabcut|rembg|auto]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phyvision.guides import guide_for  # noqa: E402
from phyvision.inference import Predictor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--mask", default=None, help="optional reference mask (white = person)")
    ap.add_argument("--seg", default="auto", choices=["auto", "rembg", "grabcut"])
    a = ap.parse_args()
    r = Predictor().predict_file(a.image, a.mask, a.seg)
    print(f"\nPhyVision result: {r['label_display']}  ({r['confidence'] * 100:.1f}% confidence)")
    for c, p in sorted(r["probabilities"].items(), key=lambda kv: -kv[1]):
        print(f"  {c:9s} {'#' * int(p * 40):40s} {p * 100:5.1f}%")
    print("Segmentation:", r["segmentation"])
    print("Proportions:", json.dumps(r["metrics"]))
    for w in r["warnings"]:
        print("Note:", w)
    if r["training_image_match"]:
        print("Integrity: this file is byte-identical to a training image (label:", r["training_image_match"], ")")
    g = guide_for(r["label"])
    print(f"\nFitness guide: {g['title']} - {g['tagline']}\n{g['summary']}")
    for t in g["goals"]:
        print("  -", t)


if __name__ == "__main__":
    main()

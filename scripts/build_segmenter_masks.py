"""Cut every selected dataset image out *from scratch* with the real segmenters
(rembg and GrabCut) and save the masks next to the images:

    <name>_seg-rembg.png   <name>_seg-grabcut.png

PhyNet is fine-tuned on these so it learns the typical mistakes each segmenter
makes on photographs (extra background, missing limbs, merged arms).
The test split is never processed here.

    python scripts/build_segmenter_masks.py --method grabcut --split train --start 0 --count 500
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402

from phyvision.config import CLASSES, DATA_DIR  # noqa: E402
from phyvision.silhouette import _mask_grabcut, _mask_rembg, clean_mask, rembg_available  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["rembg", "grabcut"], required=True)
    ap.add_argument("--split", choices=["train", "val"], required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=500)
    a = ap.parse_args()
    if a.method == "rembg" and not rembg_available():
        raise SystemExit("rembg is not installed: pip install -r requirements.txt")
    seg = _mask_rembg if a.method == "rembg" else _mask_grabcut
    t0, n, skipped = time.time(), 0, 0
    for c in CLASSES:
        files = sorted((DATA_DIR / a.split / c).glob("*.jpg"))[a.start:a.start + a.count]
        for f in files:
            out = f.with_name(f.stem + f"_seg-{a.method}.png")
            if out.exists():
                continue
            img = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
            raw = seg(img)
            if raw is None or raw.mean() < 2:  # segmenter found (almost) nobody: skip
                skipped += 1
                continue
            m = clean_mask(raw)
            cv2.imwrite(str(out), m, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            n += 1
        print(f"[{a.method}/{a.split}] {c} done, {n} masks, {skipped} skipped, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

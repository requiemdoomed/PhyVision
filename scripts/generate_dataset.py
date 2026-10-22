"""Generate the PhyVision synthetic physique dataset.

Creates  data/phyvision_dataset/{train,val,test}/{athletic,normal,obese,skinny}/
with one JPEG per person, a matching *_mask.png segmentation mask, and a
metadata.csv per split describing every rendered body.

Splits use disjoint random seeds, so no person appears in more than one split.

Usage:
    python scripts/generate_dataset.py                  # default 2000/1000/1000 per class
    python scripts/generate_dataset.py --train 300 --val 100 --test 100   # quick version
"""
import argparse
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402

from phyvision.config import CLASSES, DATA_DIR, IMG_H, SPLIT_SIZES  # noqa: E402
from phyvision.synth import generate_person  # noqa: E402

SPLIT_SEED = {"train": 0, "val": 10_000_000, "test": 20_000_000}
CLASS_SEED = {c: i * 1_000_000 for i, c in enumerate(CLASSES)}
FIELDS = ["file", "label", "sex", "framing", "visible", "arm_pose", "arm_left", "arm_right", "turn",
          "top", "bottom", "background", "width", "height", "shoulder_waist_ratio", "waist_hip_ratio", "seed"]


def _job(args):
    split, label, idx, out_dir = args
    seed = SPLIT_SEED[split] + CLASS_SEED[label] + idx
    img, mask, meta = generate_person(label, seed, out_h=IMG_H)  # width varies (portrait 0.62-0.95)
    name = f"{label}_{split}_{idx:05d}"
    cv2.imwrite(str(out_dir / f"{name}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(str(out_dir / f"{name}_mask.png"), mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    row = {k: meta.get(k, "") for k in FIELDS}
    row.update(file=f"{label}/{name}.jpg", seed=seed, width=img.shape[1], height=img.shape[0])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=SPLIT_SIZES["train"])
    ap.add_argument("--val", type=int, default=SPLIT_SIZES["val"])
    ap.add_argument("--test", type=int, default=SPLIT_SIZES["test"])
    ap.add_argument("--out", type=Path, default=DATA_DIR)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()

    sizes = {"train": a.train, "val": a.val, "test": a.test}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for split, n in sizes.items():
            jobs = []
            for label in CLASSES:
                d = a.out / split / label
                d.mkdir(parents=True, exist_ok=True)
                jobs += [(split, label, i, d) for i in range(n)]
            rows = list(pool.map(_job, jobs, chunksize=32))
            with open(a.out / split / "metadata.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(rows)
            print(f"[{split}] {len(rows)} images ({n} per class)  t={time.time() - t0:.0f}s", flush=True)
    print("Dataset ready at", a.out)


if __name__ == "__main__":
    main()

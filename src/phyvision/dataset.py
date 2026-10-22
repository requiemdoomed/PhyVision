"""PyTorch dataset over the PhyVision silhouette masks.

Training masks are corrupted on the fly to mimic real-world segmentation
errors (eroded/dilated borders, ragged edges, missing feet, stray blobs,
holes, small rotations, flips). This is what lets a model trained on
rendered people cope with masks cut out of real photographs.
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import CLASSES, DATA_DIR
from .silhouette import normalise, shape_profile


def list_split(split: str, root: Path = DATA_DIR, kind: str = "reference"):
    """kind='reference' -> exact masks; kind='segmenter' -> masks cut out by
    rembg / GrabCut (see scripts/build_segmenter_masks.py)."""
    pattern = "*_mask.png" if kind == "reference" else "*_seg-*.png"
    items = []
    for ci, c in enumerate(CLASSES):
        for p in sorted((root / split / c).glob(pattern)):
            items.append((p, ci))
    return items


def load_visible(split: str, root: Path = DATA_DIR) -> dict:
    """image stem -> visible body fraction (from metadata.csv)."""
    out = {}
    p = root / split / "metadata.csv"
    if p.exists():
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    out[Path(r["file"]).stem] = float(r.get("visible") or 1.0)
                except ValueError:
                    out[Path(r["file"]).stem] = 1.0
    return out


def _stem(mask_path: Path) -> str:
    n = mask_path.stem
    return n.split("_seg-")[0] if "_seg-" in n else n[: -len("_mask")] if n.endswith("_mask") else n


def augment_mask(m: np.ndarray, rng: np.random.Generator):
    """Returns (augmented mask, fraction of the visible body height kept)."""
    keep = 1.0
    m = (m > 127).astype(np.uint8)
    h, w = m.shape
    if rng.random() < 0.5:
        m = m[:, ::-1].copy()
    if rng.random() < 0.6:  # border error of the segmenter
        k = int(rng.integers(1, 4))
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        m = cv2.dilate(m, ker) if rng.random() < 0.5 else cv2.erode(m, ker)
    if rng.random() < 0.4:  # ragged edges
        f = cv2.GaussianBlur(m.astype(np.float32), (0, 0), rng.uniform(0.8, 2.0))
        f += rng.normal(0, 0.12, f.shape).astype(np.float32)
        m = (f > 0.5).astype(np.uint8)
    if rng.random() < 0.25:  # bottom of the body cut by the frame
        ys = np.nonzero(m.any(1))[0]
        if len(ys) and ys.max() > ys.min():
            frac = rng.uniform(0.0, 0.15)
            cut = int(ys.max() - (ys.max() - ys.min()) * frac)
            m[cut:] = 0
            keep = 1.0 - frac
    if rng.random() < 0.2:  # stray background blob touching the person
        ys, xs = np.nonzero(m)
        if len(ys):
            i = rng.integers(len(ys))
            r = int(rng.uniform(0.01, 0.04) * h)
            cv2.circle(m, (int(xs[i]), int(ys[i])), r, 1, -1)
    if rng.random() < 0.2:  # holes
        ys, xs = np.nonzero(m)
        for _ in range(int(rng.integers(1, 4))):
            i = rng.integers(len(ys))
            cv2.circle(m, (int(xs[i]), int(ys[i])), int(rng.uniform(0.005, 0.02) * h), 0, -1)
    if rng.random() < 0.5:  # camera roll / slight perspective
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.normal(0, 3), 1.0)
        M[0, 0] *= rng.uniform(0.97, 1.03)
        m = cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_NEAREST)
    return m * 255, keep


def to_tensors(mask: np.ndarray):
    sil = normalise(mask)
    prof = shape_profile(mask)
    if sil is None or prof is None:
        return None
    sil_t = torch.from_numpy((sil - 0.5) / 0.5).unsqueeze(0)
    return sil_t, torch.from_numpy(prof)


class SilhouetteDataset(Dataset):
    def __init__(self, split: str, augment: bool = False, root: Path = DATA_DIR, seed: int = 0,
                 kind: str = "reference"):
        self.items = list_split(split, root, kind)
        if not self.items:
            raise FileNotFoundError(f"No masks found in {root / split}. Run scripts/generate_dataset.py")
        self.augment = augment
        self.seed = seed
        self.epoch = 0
        # Cache masks in RAM (uint8) for speed.
        self.masks = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p, _ in self.items]
        self.labels = [c for _, c in self.items]
        vis = load_visible(split, root)
        self.visible = [vis.get(_stem(p), 1.0) for p, _ in self.items]
        if not augment or kind != "reference":
            # pre-compute deterministic tensors; drop masks where segmentation found nobody
            cache = [to_tensors(m) for m in self.masks]
            keep = [i for i, t in enumerate(cache) if t is not None]
            self.items = [self.items[i] for i in keep]
            self.masks = [self.masks[i] for i in keep]
            self.labels = [self.labels[i] for i in keep]
            self.visible = [self.visible[i] for i in keep]
            self.cache = [cache[i] for i in keep]

    def __len__(self):
        return len(self.items)

    def set_epoch(self, e: int):
        self.epoch = e

    def __getitem__(self, i):
        v = self.visible[i]
        if not self.augment:
            sil, prof = self.cache[i]
        else:
            rng = np.random.default_rng((self.seed, self.epoch, i))
            m, keep = augment_mask(self.masks[i], rng)
            out = to_tensors(m)
            if out is None:
                out, keep = to_tensors(self.masks[i]), 1.0
            sil, prof = out
            v = v * keep
        return sil, prof, self.labels[i], torch.tensor(v, dtype=torch.float32)

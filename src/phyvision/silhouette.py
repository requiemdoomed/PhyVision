"""Silhouette extraction, normalisation and body-shape descriptors.

PhyVision classifies physique from the *shape* of the person, not from their
skin, face or clothes colour. Every image therefore goes through:

    photo -> person mask -> cleaned silhouette -> (128x128 tensor, shape profile)

Mask sources, in priority order:
  1. The image's own alpha channel (transparent PNG cut-outs)
  2. rembg with the `u2net_human_seg` model (best for real photos, optional)
  3. OpenCV GrabCut (offline fallback, works best on plain backgrounds)
"""
from __future__ import annotations

import numpy as np
import cv2

from .config import SIL_SIZE, PROFILE_BINS

_REMBG_SESSION = None
_REMBG_STATE = None  # None = untested, True = available, False = unavailable


# Segmentation
def rembg_available() -> bool:
    global _REMBG_STATE
    if _REMBG_STATE is None:
        try:
            import rembg  # noqa: F401
            _REMBG_STATE = True
        except Exception:
            _REMBG_STATE = False
    return _REMBG_STATE


def _mask_rembg(rgb: np.ndarray) -> np.ndarray | None:
    global _REMBG_SESSION, _REMBG_STATE
    if not rembg_available():
        return None
    try:
        from rembg import remove, new_session
        if _REMBG_SESSION is None:
            _REMBG_SESSION = new_session("u2net_human_seg")
        out = remove(rgb, session=_REMBG_SESSION, only_mask=True)
        out = np.asarray(out)
        if out.ndim == 3:
            out = out[..., -1]
        return (out > 127).astype(np.uint8) * 255
    except Exception:
        _REMBG_STATE = False  # e.g. model download blocked; don't retry every call
        return None


def _grabcut_once(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    mask = np.zeros((h, w), np.uint8)
    rect = (max(1, int(w * 0.04)), max(1, int(h * 0.015)), int(w * 0.92), int(h * 0.97))
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, rect, bgd, fgd, 6, cv2.GC_INIT_WITH_RECT)
    return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)


def _mask_grabcut(rgb: np.ndarray) -> np.ndarray:
    """Two-pass GrabCut: a loose first pass, then a second pass restricted to a
    tight box around the first result (everything outside is background)."""
    h, w = rgb.shape[:2]
    scale = 360.0 / max(h, w) if max(h, w) > 360 else 1.0
    small = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale != 1.0 else rgb
    sh, sw = small.shape[:2]
    first = clean_mask(_grabcut_once(small))
    ys, xs = np.nonzero(first)
    fg = first
    if len(ys):
        mask = np.full((sh, sw), cv2.GC_BGD, np.uint8)
        px, py = int(0.06 * sw), int(0.02 * sh)
        mask[max(0, ys.min() - py):min(sh, ys.max() + py), max(0, xs.min() - px):min(sw, xs.max() + px)] = cv2.GC_PR_BGD
        mask[first > 0] = cv2.GC_PR_FGD
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), mask, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
            second = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
            if second.sum() > 0.2 * first.sum():
                fg = second
        except cv2.error:
            pass
    if scale != 1.0:
        fg = cv2.resize(fg, (w, h), interpolation=cv2.INTER_NEAREST)
    return fg


def extract_mask(img: np.ndarray, method: str = "auto") -> tuple[np.ndarray, str]:
    """img: HxWx3 RGB or HxWx4 RGBA uint8. Returns (mask 0/255, method used)."""
    if img.ndim == 3 and img.shape[2] == 4:
        a = img[..., 3]
        if (a < 250).mean() > 0.05:  # meaningful transparency => already cut out
            return clean_mask((a > 127).astype(np.uint8) * 255), "alpha"
        img = img[..., :3]
    if method in ("auto", "rembg"):
        m = _mask_rembg(img)
        if m is not None and m.mean() > 2:
            return clean_mask(m), "rembg"
    return clean_mask(_mask_grabcut(img)), "grabcut"


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """Keep the person (largest blob + attached limbs), fill holes, smooth edges."""
    m = (mask > 127).astype(np.uint8)
    if m.sum() == 0:
        return m * 255
    k = max(3, int(round(min(m.shape) * 0.008)) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        big = areas.max()
        keep = [i + 1 for i, a in enumerate(areas) if a >= 0.06 * big]
        m = np.isin(lab, keep).astype(np.uint8)
    # fill internal holes
    inv = (1 - m).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, 4)
    h, w = m.shape
    for i in range(1, n):
        x, y, bw, bh, a = stats[i]
        touches = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        if not touches and a < 0.02 * h * w:
            m[lab == i] = 1
    return m * 255


# Normalisation + descriptors
def crop_to_person(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.nonzero(mask > 127)
    if len(ys) < 50:
        return None
    return (mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1] > 127).astype(np.uint8)


def normalise(mask: np.ndarray, size: int = SIL_SIZE) -> np.ndarray | None:
    """Crop the silhouette, scale so the body height spans 94% of a square
    canvas (preserving aspect ratio) and centre it. Returns float32 0..1."""
    crop = crop_to_person(mask)
    if crop is None:
        return None
    h, w = crop.shape
    target = size * 0.94
    s = min(target / h, target / w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    res = cv2.resize(crop.astype(np.float32), (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), np.float32)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = res
    return canvas


def _central_run(row: np.ndarray, cx: int) -> int:
    idx = np.nonzero(row)[0]
    if len(idx) == 0:
        return 0
    c = idx[np.argmin(np.abs(idx - cx))]
    l = c
    while l - 1 >= 0 and row[l - 1]:
        l -= 1
    r = c
    while r + 1 < len(row) and row[r + 1]:
        r += 1
    return r - l + 1


def shape_profile(mask: np.ndarray, bins: int = PROFILE_BINS) -> np.ndarray | None:
    """Height-normalised width profile: for each of `bins` horizontal bands we
    record total foreground width and the width of the central (torso/leg) run,
    plus 4 global shape statistics. Length = 2*bins + 4."""
    crop = crop_to_person(mask)
    if crop is None:
        return None
    h, w = crop.shape
    cx = int(np.round(np.nonzero(crop)[1].mean()))
    total = np.zeros(bins, np.float32)
    central = np.zeros(bins, np.float32)
    edges = np.linspace(0, h, bins + 1).astype(int)
    for i in range(bins):
        a, b = edges[i], max(edges[i] + 1, edges[i + 1])
        rows = crop[a:b]
        total[i] = rows.sum(1).mean() / h
        central[i] = np.mean([_central_run(r, cx) for r in rows]) / h
    area = crop.sum()
    cnts, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull_area = sum(cv2.contourArea(cv2.convexHull(c)) for c in cnts) or 1.0
    glob = np.array([w / h, area / (w * h), area / (h * h) * 4, min(1.0, area / hull_area)], np.float32)
    return np.concatenate([total * 4, central * 4, glob]).astype(np.float32)


def body_metrics(mask: np.ndarray, visible: float = 1.0) -> dict:
    """Human-readable proportion estimates from the silhouette (2-D, frontal).

    `visible` is the fraction of the body in the photo (from the framing head;
    1.0 = head to feet). Measurement heights are given as fractions of full
    body height and converted to the visible part, so cropped photos are
    measured at the right places. Widths are reported relative to the
    estimated full body height. If an arm overlaps the outline the numbers
    can be implausible; then `reliable` is False and the UI hides them."""
    prof = shape_profile(mask, bins=100)
    if prof is None:
        return {}
    central = prof[100:200] / 4          # central-run width / visible height
    v = float(min(1.0, max(0.4, visible)))

    def band(a, b, fn):
        lo, hi = int(a / v * 100), int(b / v * 100)
        seg = central[min(lo, 99):min(max(hi, lo + 1), 100)]
        seg = seg[seg > 0]
        return float(fn(seg)) * v if len(seg) else 0.0   # -> fraction of full body height

    shoulder = band(0.18, 0.22, np.median)
    waist = band(0.34, 0.44, np.min)
    hip = band(0.45, 0.53, np.max) if 0.53 / v <= 1.0 else 0.0
    crop = crop_to_person(mask)
    h, w = crop.shape
    s2w = shoulder / waist if waist else None
    w2h = waist / hip if hip else None
    # Arms must be visibly separate from the torso at waist level (three
    # separate runs: arm, torso, arm), otherwise an arm is merged into the
    # waist outline and the widths are not trustworthy.
    hh = crop.shape[0]
    rows = range(int(0.34 / v * hh), min(hh, int(0.44 / v * hh) + 1))
    def runs(r):
        d = np.diff(np.concatenate([[0], r.astype(np.int8), [0]]))
        return int((d == 1).sum())
    sep = [runs(crop[y]) >= 3 for y in rows]
    arms_free = bool(sep) and float(np.mean(sep)) >= 0.6
    reliable = bool(arms_free and s2w and 0.6 <= s2w <= 2.6 and 0.12 <= shoulder <= 0.40
                    and (w2h is None or 0.5 <= w2h <= 1.6))
    return {
        "shoulder_width_pct_height": round(shoulder * 100, 1),
        "waist_width_pct_height": round(waist * 100, 1),
        "hip_width_pct_height": round(hip * 100, 1) if hip else None,
        "shoulder_to_waist": round(s2w, 2) if s2w else None,
        "waist_to_hip": round(w2h, 2) if w2h else None,
        "silhouette_fill": round(float(crop.mean()), 3),
        "aspect_w_h": round(w / h, 3),
        "arms_separate": arms_free,
        "reliable": reliable,
    }


def overlay(rgb: np.ndarray, mask: np.ndarray, colour=(56, 189, 248)) -> np.ndarray:
    """Tint the detected person and draw its outline (for the UI)."""
    out = rgb.copy().astype(np.float32)
    m = mask > 127
    out[m] = out[m] * 0.55 + np.array(colour, np.float32) * 0.45
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    t = max(2, int(round(min(mask.shape) * 0.006)))
    cv2.drawContours(out, cnts, -1, (255, 255, 255), t, cv2.LINE_AA)
    return np.clip(out, 0, 255).astype(np.uint8)

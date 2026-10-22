"""High-level prediction API used by the web app, notebook and CLI.

    from phyvision.inference import Predictor
    p = Predictor()
    result = p.predict_file("photo.jpg")
"""
from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from .config import CHECKPOINT, CLASSES, CLASS_LABELS
from .dataset import to_tensors
from .integrity import load_hash_db, md5_bytes
from .model import PhyNet
from .silhouette import body_metrics, extract_mask, clean_mask
from .synth import framing_name

UPSCALE_BELOW = 600  # px; photos shorter than this are enlarged before segmentation

FRAMING_LABELS = {"full": "Head to feet", "knees": "Head to knees",
                  "thigh": "Head to mid-thigh", "waist": "Head to waist"}


def load_image_bytes(data: bytes) -> np.ndarray:
    """Decode bytes to RGB/RGBA uint8, honouring EXIF rotation from phones."""
    im = Image.open(io.BytesIO(data))
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
    else:
        im = im.convert("RGB")
    arr = np.asarray(im)
    # keep processing fast for huge phone photos
    h, w = arr.shape[:2]
    s = 1024 / max(h, w)
    if s < 1:
        arr = cv2.resize(arr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return arr


class Predictor:
    def __init__(self, checkpoint: Path = CHECKPOINT, device: str = "cpu"):
        ck = torch.load(checkpoint, map_location=device, weights_only=False)
        self.classes = ck.get("classes", CLASSES)
        self.model = PhyNet(num_classes=len(self.classes))
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval().to(device)
        self.device = device
        self.meta = {k: ck.get(k) for k in ("version", "epoch", "val_acc")}
        self.hash_db = load_hash_db()

    @torch.no_grad()
    def predict_mask(self, mask: np.ndarray) -> tuple[np.ndarray, float]:
        """(class probabilities, visible body fraction) for a silhouette,
        averaged over the silhouette and its mirror image."""
        probs, fr = [], []
        for m in (mask, mask[:, ::-1].copy()):
            t = to_tensors(m)
            if t is None:
                raise ValueError("No person could be found in this image. Use a photo where one person "
                                 "is clearly visible against the background.")
            sil, prof = t
            logits, vis = self.model(sil.unsqueeze(0).to(self.device), prof.unsqueeze(0).to(self.device))
            probs.append(torch.softmax(logits, 1)[0].cpu().numpy())
            fr.append(float(vis[0]))
        return np.mean(probs, 0), float(np.mean(fr))

    def predict_array(self, img: np.ndarray, mask: np.ndarray | None = None,
                      seg_method: str = "auto", raw_bytes: bytes | None = None) -> dict:
        orig_hw = img.shape[:2]
        if mask is not None:
            mask, method = clean_mask(mask), "reference"
        else:
            # Small photos give jagged cut-outs. Enlarge them before segmentation
            # so the outline is smoother; the photo check still reports the
            # original size.
            h = img.shape[0]
            if h < UPSCALE_BELOW:
                f = min(4.0, UPSCALE_BELOW / h)
                img = cv2.resize(img, (round(img.shape[1] * f), round(h * f)), interpolation=cv2.INTER_CUBIC)
            mask, method = extract_mask(img, seg_method)
        rgb = img[..., :3] if img.ndim == 3 else np.dstack([img] * 3)
        if (mask > 127).sum() < 0.01 * mask.size:
            raise ValueError("No person could be found in this image. Use a photo where one person "
                             "is clearly visible against the background.")
        probs, visible = self.predict_mask(mask)
        checks = photo_checks(rgb, mask, visible, orig_hw)
        warnings = [c["detail"] for c in checks if c["status"] != "ok"]
        order = np.argsort(probs)[::-1]
        top = self.classes[order[0]]
        result = {
            "label": top,
            "label_display": CLASS_LABELS.get(top, top.title()),
            "confidence": float(probs[order[0]]),
            "probabilities": {c: float(p) for c, p in zip(self.classes, probs)},
            "runner_up": self.classes[order[1]],
            "borderline": bool(probs[order[0]] < 0.6),
            "segmentation": method,
            "metrics": body_metrics(mask, visible),
            "framing": {"visible": round(visible, 3), "name": framing_name(visible),
                        "label": FRAMING_LABELS[framing_name(visible)]},
            "checks": checks,
            "warnings": warnings,
            "training_image_match": None,
        }
        if raw_bytes is not None:
            hit = self.hash_db.get(md5_bytes(raw_bytes))
            if hit:
                result["training_image_match"] = hit
        result["_mask"] = mask
        result["_rgb"] = rgb
        return result

    def predict_bytes(self, data: bytes, mask: np.ndarray | None = None, seg_method: str = "auto") -> dict:
        return self.predict_array(load_image_bytes(data), mask, seg_method, raw_bytes=data)

    def predict_file(self, path, mask_path=None, seg_method: str = "auto") -> dict:
        data = Path(path).read_bytes()
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path else None
        return self.predict_bytes(data, mask, seg_method)


def photo_checks(rgb: np.ndarray, mask: np.ndarray, visible: float, orig_hw=None) -> list[dict]:
    """Compare an uploaded photo with the PhyVision photo guidelines.
    Returns a list of {id, name, status ('ok' | 'warn' | 'bad'), detail}."""
    H, W = mask.shape
    H0, W0 = orig_hw or (H, W)  # original upload size, for the resolution check
    m = mask > 127
    ys, xs = np.nonzero(m)
    out = []

    def add(cid, name, status, detail):
        out.append({"id": cid, "name": name, "status": status, "detail": detail})

    # resolution
    if min(H0, W0) < 160:
        add("resolution", "Resolution", "bad", f"The photo is very small ({W0} x {H0} px). Use one at least 400 px tall.")
    elif H0 < 400:
        add("resolution", "Resolution", "warn", f"The photo is small ({W0} x {H0} px). 600 px or taller gives cleaner outlines.")
    else:
        add("resolution", "Resolution", "ok", f"{W0} x {H0} px.")

    # aspect ratio
    ar = W / H
    if ar > 1.15:
        add("aspect", "Orientation", "bad", f"Landscape photo (width:height {ar:.2f}). Use a portrait photo, about 2:3 or 3:4.")
    elif ar > 0.95:
        add("aspect", "Orientation", "warn", f"Square-ish photo (width:height {ar:.2f}). Portrait 2:3 or 3:4 works best.")
    else:
        add("aspect", "Orientation", "ok", f"Portrait, width:height {ar:.2f}.")

    # people in the photo
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    areas = sorted(stats[1:, cv2.CC_STAT_AREA], reverse=True)
    if len(areas) > 1 and areas[1] > 0.25 * areas[0]:
        add("single", "One person", "bad", "More than one person or object was detected. Crop the photo to one person.")
    else:
        add("single", "One person", "ok", "One person detected.")

    # background separation
    fill = m.mean()
    if fill > 0.70:
        add("background", "Background", "bad", "Most of the photo was detected as the person, so the background was not separated. Use a plain background.")
    else:
        add("background", "Background", "ok", "Person separated from the background.")

    # body height in frame and head room
    bh = ys.max() - ys.min() + 1
    frac = bh / H
    if frac < 0.5:
        add("size", "Person size", "bad", f"The person fills only {frac * 100:.0f}% of the photo height. Move closer or crop so they fill 75-95%.")
    elif frac < 0.7:
        add("size", "Person size", "warn", f"The person fills {frac * 100:.0f}% of the photo height. 75-95% is ideal.")
    else:
        add("size", "Person size", "ok", f"The person fills {frac * 100:.0f}% of the photo height.")
    if ys.min() <= 1:
        add("head", "Head in frame", "bad", "The top of the head is cut off. Leave a little space above the head.")
    else:
        add("head", "Head in frame", "ok", "Head fully visible.")

    # centring
    cx = xs.mean() / W
    if abs(cx - 0.5) > 0.22:
        add("centre", "Centred", "warn", "The person is near the edge of the photo. Centre them for best results.")
    elif xs.min() <= 1 or xs.max() >= W - 2:
        add("centre", "Centred", "warn", "Part of the body touches the side of the photo. Keep arms and shoulders inside the frame.")
    else:
        add("centre", "Centred", "ok", "Person centred, sides of the body in frame.")

    # framing (from the model)
    fn = framing_name(visible)
    if fn == "full":
        add("framing", "Framing", "ok", "Full body visible, head to feet. This is the most reliable framing.")
    elif fn in ("knees", "thigh"):
        add("framing", "Framing", "warn", f"{FRAMING_LABELS[fn]} visible (about {visible * 100:.0f}% of the body). "
            "Supported, but a head-to-feet photo is more reliable.")
    else:
        add("framing", "Framing", "bad", "Only the upper body is visible. Include at least down to mid-thigh, ideally head to feet.")
    return out

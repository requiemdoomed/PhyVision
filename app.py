"""PhyVision web server.

Serves the PhyVision interface (index.html, web/) and a small JSON API that
runs the trained PhyNet model on uploaded photos.

    pip install -r requirements.txt
    python app.py                 # then open http://127.0.0.1:5000
"""
from __future__ import annotations

import argparse
import base64
import json
import random
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from flask import Flask, abort, jsonify, request, send_from_directory  # noqa: E402

from phyvision import config  # noqa: E402
from phyvision.guides import load_guides  # noqa: E402
from phyvision.integrity import md5_bytes  # noqa: E402
from phyvision.silhouette import normalise, overlay, rembg_available  # noqa: E402

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB uploads

_predictor = None
_predictor_error = None
_lock = threading.Lock()
_started = time.time()

PUBLIC_DIRS = {"web", "reports", "docs", "notebooks", "legacy", "data", "src", "scripts", "models"}
PUBLIC_ROOT_FILES = {"index.html", "README.md", "LICENSE", "requirements.txt", "app.py", "CITATION.cff"}


def get_predictor():
    global _predictor, _predictor_error
    if _predictor is None and _predictor_error is None:
        try:
            from phyvision.inference import Predictor
            _predictor = Predictor()
        except FileNotFoundError:
            _predictor_error = "Model checkpoint not found. Run: python scripts/train.py"
        except Exception as e:  # pragma: no cover
            _predictor_error = f"Could not load model: {e}"
    return _predictor


def _png_data_url(arr: np.ndarray) -> str:
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", arr)
    return "data:image/png;base64," + base64.b64encode(buf).decode()


def _jpg_data_url(rgb: np.ndarray, max_side=720) -> str:
    h, w = rgb.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        rgb = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def _demo_index():
    p = config.DEMO_DIR / "demos.json"
    return json.loads(p.read_text()) if p.exists() else []


_DEMO_HASH = {}


def _demo_mask_for(data: bytes):
    """Uploaded demo images reuse their reference mask (exact segmentation)."""
    if not _DEMO_HASH:
        for d in _demo_index():
            f = config.WEB_DIR / d["image"]
            if f.exists():
                _DEMO_HASH[md5_bytes(f.read_bytes())] = config.WEB_DIR / d["mask"]
    m = _DEMO_HASH.get(md5_bytes(data))
    return cv2.imread(str(m), cv2.IMREAD_GRAYSCALE) if m else None


def _dataset_counts():
    out = {}
    for s in ("train", "val", "test"):
        out[s] = {c: len(list((config.DATA_DIR / s / c).glob("*.jpg"))) for c in config.CLASSES}
    return out


# API
@app.get("/api/status")
def status():
    p = get_predictor()
    metrics = json.loads(config.METRICS_JSON.read_text()) if config.METRICS_JSON.exists() else None
    return jsonify({
        "project": config.PROJECT_NAME, "version": config.VERSION, "author": config.AUTHOR,
        "model_loaded": p is not None, "model_error": _predictor_error,
        "model_meta": p.meta if p else None,
        "classes": config.CLASSES,
        "segmentation": {"rembg": rembg_available(), "grabcut": True},
        "dataset": _dataset_counts(),
        "test_accuracy": metrics["reference_masks"]["accuracy"] if metrics else None,
        "photo_pipeline_accuracy": metrics["photo_pipeline_grabcut"]["accuracy"] if metrics else None,
        "rembg_pipeline_accuracy": (metrics.get("photo_pipeline_rembg") or {}).get("accuracy") if metrics else None,
        "uptime_s": round(time.time() - _started),
    })


@app.post("/api/classify")
def classify():
    p = get_predictor()
    if p is None:
        return jsonify({"error": _predictor_error or "Model not available"}), 503
    f = request.files.get("image")
    demo_id = request.form.get("demo_id")
    seg = request.form.get("seg", "auto")
    if seg not in ("auto", "rembg", "grabcut"):
        seg = "auto"
    if f is None and demo_id:
        d = next((d for d in _demo_index() if d["id"] == demo_id), None)
        if d is None:
            return jsonify({"error": "Unknown demo"}), 404
        data = (config.WEB_DIR / d["image"]).read_bytes()
    elif f is not None:
        data = f.read()
    else:
        return jsonify({"error": "No image uploaded"}), 400
    if not data:
        return jsonify({"error": "The uploaded file is empty"}), 400

    t0 = time.time()
    try:
        with _lock:
            r = p.predict_bytes(data, mask=_demo_mask_for(data), seg_method=seg)
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        return jsonify({"error": f"Could not read that image ({type(e).__name__}). Try a JPG or PNG."}), 400

    mask, rgb = r.pop("_mask"), r.pop("_rgb")
    sil = normalise(mask)
    r["images"] = {
        "input": _jpg_data_url(rgb),
        "overlay": _jpg_data_url(overlay(rgb, mask)),
        "silhouette": _png_data_url((sil * 255).astype(np.uint8)) if sil is not None else None,
    }
    r["guide"] = load_guides()["classes"][r["label"]]
    r["elapsed_ms"] = round((time.time() - t0) * 1000)
    return jsonify(r)


@app.get("/api/demos")
def demos():
    return jsonify(_demo_index())


@app.get("/api/guides")
def guides():
    return jsonify(load_guides())


@app.get("/api/metrics")
def metrics():
    if not config.METRICS_JSON.exists():
        return jsonify({"error": "No metrics yet. Run: python scripts/evaluate.py"}), 404
    m = json.loads(config.METRICS_JSON.read_text())
    hist = config.REPORTS_DIR / "history.json"
    m["history"] = json.loads(hist.read_text()) if hist.exists() else []
    return jsonify(m)


@app.get("/api/samples")
def samples():
    split = request.args.get("split", "test")
    if split not in ("train", "val", "test"):
        abort(400)
    n = min(12, int(request.args.get("n", 6)))
    out = {}
    for c in config.CLASSES:
        files = sorted((config.DATA_DIR / split / c).glob("*.jpg"))
        pick = random.sample(files, min(n, len(files))) if files else []
        out[c] = [f"/{f.relative_to(ROOT).as_posix()}" for f in pick]
    return jsonify(out)


@app.get("/api/files")
def files():
    from_manifest = config.WEB_DIR / "assets" / "manifest.json"
    entries = json.loads(from_manifest.read_text()) if from_manifest.exists() else []
    for e in entries:
        fp = ROOT / e["path"]
        e["exists"] = fp.exists()
        if fp.is_file():
            e["size"] = fp.stat().st_size
    return jsonify(entries)


# Static files
@app.get("/")
def home():
    return send_from_directory(ROOT, "index.html")


@app.get("/<path:path>")
def static_files(path):
    top = path.split("/", 1)[0]
    if path in PUBLIC_ROOT_FILES or top in PUBLIC_DIRS:
        return send_from_directory(ROOT, path)
    abort(404)


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "That file is larger than 20 MB. Please upload a smaller image."}), 413


def main():
    ap = argparse.ArgumentParser(description="Run the PhyVision web app")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    get_predictor()
    url = f"http://{a.host}:{a.port}"
    print(f"\n  PhyVision {config.VERSION}  by {config.AUTHOR}\n  Open {url} in your browser (Ctrl+C to stop)\n")
    if _predictor_error:
        print("  WARNING:", _predictor_error)
    if not a.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    app.run(host=a.host, port=a.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()

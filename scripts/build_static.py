"""Bundle data for offline use of the HTML pages.

Writes web/assets/static_data.js (loaded with a <script> tag, so it works even
when the pages are opened directly from disk) and web/assets/manifest.json.
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phyvision import config  # noqa: E402

MANIFEST = [
    ("index.html", "Main control center: system status, model performance, dataset, files and run instructions"),
    ("web/app.html", "PhyVision analyzer: upload a photo or pick a demo person, see the class, proportions and fitness guide"),
    ("web/photo-guide.html", "How to take a photo PhyVision can read: framing, aspect ratio, pose, clothing, background, light"),
    ("web/guides.html", "Fitness guide library for the four physique classes"),
    ("app.py", "Flask web server and JSON API (/api/status, /api/classify, /api/demos, /api/metrics, ...)"),
    ("notebooks/PhyVision.ipynb", "End-to-end notebook: dataset, training, evaluation, inference and guides (runs in Jupyter or Colab)"),
    ("src/phyvision/synth.py", "Procedural human renderer that generates the AI-made physique dataset"),
    ("src/phyvision/silhouette.py", "Person segmentation (rembg / GrabCut / alpha), silhouette normalisation and shape measurements"),
    ("src/phyvision/model.py", "PhyNet: ResNet-18 silhouette branch fused with a shape-profile MLP"),
    ("src/phyvision/dataset.py", "PyTorch dataset with segmentation-error augmentation"),
    ("src/phyvision/train.py", "Stage 1 training on exact masks (AdamW, one-cycle cosine schedule, label smoothing)"),
    ("src/phyvision/finetune.py", "Stage 2 fine-tuning on masks cut out by rembg and GrabCut"),
    ("src/phyvision/evaluate.py", "Test-set evaluation and report figures"),
    ("src/phyvision/inference.py", "Predictor API used by the web app, notebook and command line"),
    ("src/phyvision/integrity.py", "MD5 training-image integrity check"),
    ("src/phyvision/guides.py", "Loads fitness guide content"),
    ("src/phyvision/config.py", "Paths, class names and sizes"),
    ("scripts/generate_dataset.py", "Builds the 16,000-image dataset"),
    ("scripts/train.py", "Trains PhyNet (stage 1)"),
    ("scripts/build_segmenter_masks.py", "Cuts dataset photos out with rembg / GrabCut for stage 2"),
    ("scripts/finetune.py", "Fine-tunes PhyNet on segmenter masks (stage 2)"),
    ("scripts/evaluate.py", "Evaluates on the test split and writes reports/"),
    ("scripts/predict.py", "Command-line classification of a single image"),
    ("scripts/make_demos.py", "Renders the named demo people"),
    ("scripts/build_hash_db.py", "Builds the MD5 training-image database"),
    ("scripts/build_static.py", "Builds offline data for the HTML pages"),
    ("web/assets/guides.json", "Fitness guide content for every class"),
    ("models/phyvision_phynet.pt", "Trained PhyNet weights"),
    ("models/train_hashes.json", "MD5 fingerprints of training images"),
    ("reports/metrics.json", "Test metrics, per-class scores and confusion matrices"),
    ("data/phyvision_dataset/", "The dataset: train / val / test folders, one per class, with masks and metadata.csv"),
    ("docs/MODEL_CARD.md", "Model card: intended use, data, metrics and limitations"),
    ("docs/DATASET_CARD.md", "Dataset card: how the images were generated"),
    ("docs/HumanPhysique_FinalReport_v4.pdf", "Original v4.0 course report (the project PhyVision grew from)"),
    ("legacy/HumanBodyIdentifier_v4_original.ipynb", "Original v4.0 Colab notebook, kept for reference"),
    ("README.md", "Project overview and setup"),
    ("LICENSE", "MIT License"),
]


def main():
    guides = json.loads((config.WEB_DIR / "assets" / "guides.json").read_text(encoding="utf-8"))
    demos_p = config.DEMO_DIR / "demos.json"
    demos = json.loads(demos_p.read_text()) if demos_p.exists() else []
    metrics = json.loads(config.METRICS_JSON.read_text()) if config.METRICS_JSON.exists() else None
    hist = config.REPORTS_DIR / "history.json"
    if metrics and hist.exists():
        metrics["history"] = json.loads(hist.read_text())
    counts = {s: {c: len(list((config.DATA_DIR / s / c).glob("*.jpg"))) for c in config.CLASSES}
              for s in ("train", "val", "test")}
    rnd = random.Random(7)
    samples = {}
    for c in config.CLASSES:
        files = sorted((config.DATA_DIR / "test" / c).glob("*.jpg"))
        samples[c] = [f.relative_to(ROOT).as_posix() for f in rnd.sample(files, min(6, len(files)))]
    manifest = [{"path": p, "role": r} for p, r in MANIFEST]
    hero = None
    hm = config.DEMO_DIR / "john_bahadur_mask.png"
    if hm.exists():
        import cv2
        import numpy as np
        m = cv2.imread(str(hm), cv2.IMREAD_GRAYSCALE) > 127
        ys = np.nonzero(m.any(1))[0]
        top, bot, H = ys.min(), ys.max(), m.shape[0]
        cx = int(np.nonzero(m)[1].mean())

        from phyvision.silhouette import _central_run

        def run(y):
            return _central_run(m[y].astype(np.uint8), cx)
        hy = lambda f: int(top + f * (bot - top))
        waist_y = min(range(hy(0.34), hy(0.46)), key=run)
        hip_y = max(range(hy(0.47), hy(0.55)), key=run)
        hero = {"shoulder": round(hy(0.205) / H, 4), "waist": round(waist_y / H, 4), "hip": round(hip_y / H, 4)}
    (config.WEB_DIR / "assets" / "manifest.json").write_text(json.dumps(manifest, indent=1))
    data = {"version": config.VERSION, "author": config.AUTHOR, "guides": guides, "demos": demos,
            "metrics": metrics, "dataset_counts": counts, "samples": samples, "manifest": manifest,
            "model_present": config.CHECKPOINT.exists(), "hero": hero}
    js = ("/* Generated by scripts/build_static.py - offline data for PhyVision pages. */\n"
          "window.PHY_STATIC = " + json.dumps(data, separators=(",", ":")) + ";\n")
    (config.WEB_DIR / "assets" / "static_data.js").write_text(js, encoding="utf-8")
    print("Wrote web/assets/static_data.js and manifest.json")


if __name__ == "__main__":
    main()

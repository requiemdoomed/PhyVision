"""Evaluate PhyNet on the held-out test split and write reports/.

Two evaluations are run:
  1. Reference silhouettes: the ground-truth mask of every test image
     (measures the classifier itself).
  2. Photo pipeline: a stratified subset of the test JPEGs is segmented from
     scratch with the offline GrabCut fallback, exactly as an uploaded photo
     would be, and then classified (measures the end-to-end system).
"""
from __future__ import annotations

import argparse
import json
import time

import cv2
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import CHECKPOINT, CLASSES, CLASS_LABELS, DATA_DIR, REPORTS_DIR, METRICS_JSON, VERSION  # noqa: E402
from .dataset import SilhouetteDataset  # noqa: E402
from .inference import Predictor  # noqa: E402

INK = "#1f2937"
PALETTE = {"athletic": "#d97706", "normal": "#059669", "obese": "#db2777", "skinny": "#0284c7"}


def _report(y, p):
    rep = classification_report(y, p, labels=list(range(len(CLASSES))), target_names=CLASSES,
                                output_dict=True, zero_division=0)
    cm = confusion_matrix(y, p, labels=list(range(len(CLASSES))))
    return {"accuracy": rep["accuracy"],
            "per_class": {c: {k: rep[c][k] for k in ("precision", "recall", "f1-score", "support")} for c in CLASSES},
            "macro_f1": rep["macro avg"]["f1-score"],
            "weighted_f1": rep["weighted avg"]["f1-score"],
            "confusion_matrix": cm.tolist(), "n": int(len(y))}


def plot_confusion(cm, title, path):
    cm = np.array(cm)
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=150)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    labels = [CLASS_LABELS[c] for c in CLASSES]
    ax.set_xticks(range(4), labels)
    ax.set_yticks(range(4), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j] * 100:.1f}%", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.55 else INK)
    ax.set_title(title, fontsize=10)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_history(path_json, path_png):
    try:
        h = json.loads(path_json.read_text())
    except Exception:
        return
    ft_path = path_json.with_name("history_finetune.json")
    ft = json.loads(ft_path.read_text())[1:] if ft_path.exists() else []
    ep = [r["epoch"] for r in h]
    n = len(ep)
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 4.1), dpi=150)
    axs[0].plot(ep, [r["train_loss"] for r in h], color="#2563eb", label="train")
    axs[0].plot(ep, [r["val_loss"] for r in h], color="#ea580c", label="validation")
    axs[0].set_title("Stage 1 loss")
    axs[1].plot(ep, [r["train_acc"] * 100 for r in h], color="#2563eb", label="train (augmented)")
    axs[1].plot(ep, [r["val_acc"] * 100 for r in h], color="#ea580c", label="validation, exact masks")
    if ft:
        fe = [n + r["epoch"] for r in ft]
        axs[1].plot([n] + fe, [h[-1]["val_acc"] * 100] + [r["val_ref_acc"] * 100 for r in ft], color="#ea580c", ls="--")
        axs[1].plot(fe, [r["val_seg_acc"] * 100 for r in ft], color="#0f8a6a", marker="o", ms=3,
                    label="validation, segmenter masks")
        axs[1].axvspan(n + 0.5, n + len(ft) + 0.5, color="#e8b931", alpha=0.15, lw=0)
        axs[1].text(n + 0.9, 0.04, "stage 2: fine-tuning\non real-segmenter masks", fontsize=7, transform=axs[1].get_xaxis_transform())
    axs[1].set_title("Accuracy (%)")
    for ax in axs:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].legend(frameon=False, fontsize=7)
    axs[1].legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    fig.tight_layout()
    fig.savefig(path_png)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo-subset", type=int, default=150, help="test JPEGs per class for the photo-pipeline eval")
    ap.add_argument("--no-rembg", action="store_true", help="skip the rembg photo-pipeline evaluation")
    a = ap.parse_args(argv)
    REPORTS_DIR.mkdir(exist_ok=True)

    pred = Predictor(CHECKPOINT)
    model = pred.model

    # 1. reference silhouettes ---------------------------------------------------
    ds = SilhouetteDataset("test", augment=False)
    ys, ps, fr_err, vis_all = [], [], [], []
    with torch.no_grad():
        for sil, prof, y, vis in DataLoader(ds, batch_size=256):
            l1, f1 = model(sil, prof)
            l2, f2 = model(torch.flip(sil, [3]), prof)
            pr = (torch.softmax(l1, 1) + torch.softmax(l2, 1)) / 2
            ys += y.tolist()
            ps += pr.argmax(1).tolist()
            fr_err += ((f1 + f2) / 2 - vis).abs().tolist()
            vis_all += vis.tolist()
    ref = _report(ys, ps)
    ref["framing_mae"] = float(np.mean(fr_err))
    print(f"Reference-mask test accuracy: {ref['accuracy'] * 100:.2f}% on {ref['n']} images "
          f"(framing MAE {ref['framing_mae']:.3f})")
    from .synth import framing_name
    by_framing = {}
    for name in ("full", "knees", "thigh", "waist"):
        idx = [i for i, v in enumerate(vis_all) if framing_name(v) == name]
        if idx:
            by_framing[name] = {"n": len(idx), "accuracy": float(np.mean([ys[i] == ps[i] for i in idx]))}
    ref["by_framing"] = by_framing
    print("  by framing:", {k: f"{v['accuracy'] * 100:.1f}% (n={v['n']})" for k, v in by_framing.items()})

    # 2. photo pipeline (segmentation from scratch) ---------------------------------
    from .silhouette import rembg_available
    methods = ["grabcut"] + (["rembg"] if rembg_available() and not a.no_rembg else [])
    pipelines = {}
    from .dataset import load_visible
    test_vis = load_visible("test")
    for method in methods:
        yy, pp, vv, t0 = [], [], [], time.time()
        for ci, c in enumerate(CLASSES):
            files = sorted((DATA_DIR / "test" / c).glob("*.jpg"))[: a.photo_subset]
            for f in files:
                img = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
                try:
                    r = pred.predict_array(img, seg_method=method)
                    pl = CLASSES.index(r["label"])
                except ValueError:  # segmenter found nobody: count as an error
                    pl = -1
                yy.append(ci)
                pp.append(pl)
                vv.append(test_vis.get(f.stem, 1.0))
        pipelines[method] = _report(yy, [p if p >= 0 else (y + 1) % 4 for p, y in zip(pp, yy)])
        pipelines[method]["no_person_found"] = int(sum(p < 0 for p in pp))
        pipelines[method]["by_framing"] = {}
        for name in ("full", "knees", "thigh", "waist"):
            idx = [i for i, v in enumerate(vv) if framing_name(v) == name]
            if idx:
                pipelines[method]["by_framing"][name] = {"n": len(idx), "accuracy": float(np.mean([yy[i] == pp[i] for i in idx]))}
        print(f"Photo-pipeline ({method}) accuracy: {pipelines[method]['accuracy'] * 100:.2f}% on "
              f"{pipelines[method]['n']} images ({time.time() - t0:.0f}s)", flush=True)
        print("  by framing:", {k: f"{v['accuracy'] * 100:.1f}% (n={v['n']})" for k, v in pipelines[method]["by_framing"].items()})
    photo = pipelines["grabcut"]

    counts = {s: {c: len(list((DATA_DIR / s / c).glob("*.jpg"))) for c in CLASSES} for s in ("train", "val", "test")}
    metrics = {"version": VERSION, "model": "PhyNet (ResNet-18 silhouette branch + shape-profile MLP + framing head)",
               "checkpoint_epoch": pred.meta.get("epoch"), "best_val_acc": pred.meta.get("val_acc"),
               "dataset_counts": counts, "reference_masks": ref, "photo_pipeline_grabcut": photo,
               "photo_pipeline_rembg": pipelines.get("rembg")}
    METRICS_JSON.write_text(json.dumps(metrics, indent=2))
    plot_confusion(ref["confusion_matrix"], f"Test set, reference masks (acc {ref['accuracy'] * 100:.1f}%)",
                   REPORTS_DIR / "confusion_matrix.png")
    plot_confusion(photo["confusion_matrix"], f"Test subset, GrabCut photo pipeline (acc {photo['accuracy'] * 100:.1f}%)",
                   REPORTS_DIR / "confusion_matrix_photo_pipeline.png")
    if pipelines.get("rembg"):
        plot_confusion(pipelines["rembg"]["confusion_matrix"],
                       f"Test subset, rembg photo pipeline (acc {pipelines['rembg']['accuracy'] * 100:.1f}%)",
                       REPORTS_DIR / "confusion_matrix_rembg_pipeline.png")
    plot_history(REPORTS_DIR / "history.json", REPORTS_DIR / "training_curves.png")
    print("Wrote", METRICS_JSON)


if __name__ == "__main__":
    main()

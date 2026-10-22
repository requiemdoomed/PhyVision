"""Stage 2: fine-tune PhyNet on masks produced by the real segmenters.

Stage 1 (train.py) teaches the model body shape from exact masks. Stage 2
mixes in masks that rembg and GrabCut cut out of the dataset photos, so the
model also learns what those segmenters get wrong on real images. Model
selection uses both exact and segmenter masks from the validation split.
The test split is never touched.

    python scripts/finetune.py --epochs 6

AI assistance: this fine-tuning pipeline was developed with help from
Claude, an AI model by Anthropic, for educational purposes (learning how to
train a vision model).
"""
from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

from .config import CHECKPOINT, CLASSES, REPORTS_DIR, VERSION
from .dataset import SilhouetteDataset
from .model import PhyNet
from .train import run_epoch


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--append", action="store_true",
                    help="continue the epoch numbering of an earlier fine-tuning round in history_finetune.json")
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed)

    ck = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = PhyNet()
    model.load_state_dict(ck["state_dict"])

    ref_tr = SilhouetteDataset("train", augment=True, seed=a.seed)
    seg_tr = SilhouetteDataset("train", augment=True, seed=a.seed + 1, kind="segmenter")
    # repeat segmenter masks as needed so they make up about half of each epoch
    reps = max(1, round(len(ref_tr) / max(1, len(seg_tr))))
    train = ConcatDataset([ref_tr] + [seg_tr] * reps)
    ref_va = SilhouetteDataset("val", kind="reference")
    seg_va = SilhouetteDataset("val", kind="segmenter")
    print(f"train: {len(ref_tr)} reference + {len(seg_tr)} segmenter masks | "
          f"val: {len(ref_va)} reference + {len(seg_va)} segmenter", flush=True)

    tl = DataLoader(train, batch_size=a.batch, shuffle=True, num_workers=1, persistent_workers=True, drop_last=True)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(tl),
                                                pct_start=0.1, anneal_strategy="cos")

    def val_scores():
        _, ra = run_epoch(model, DataLoader(ref_va, batch_size=256), crit)
        _, sa = run_epoch(model, DataLoader(seg_va, batch_size=256), crit)
        return ra, sa

    ra, sa = val_scores()
    best = (ra + sa) / 2
    print(f"start      val reference {ra:.4f}  val segmenter {sa:.4f}", flush=True)
    hist_path = REPORTS_DIR / "history_finetune.json"
    history = [dict(epoch=0, val_ref_acc=ra, val_seg_acc=sa)]
    offset = 0
    if a.append and hist_path.exists():
        history = json.loads(hist_path.read_text())
        offset = history[-1]["epoch"]
    for ep in range(offset + 1, offset + a.epochs + 1):
        for d in (ref_tr, seg_tr):
            d.set_epoch(100 + ep)
        t0 = time.time()
        trl, tra = run_epoch(model, tl, crit, opt, sched)
        ra, sa = val_scores()
        score = (ra + sa) / 2
        flag = ""
        if score >= best:
            best = score
            torch.save({"state_dict": model.state_dict(), "classes": CLASSES, "version": VERSION,
                        "epoch": ck.get("epoch"), "finetune_epoch": ep, "val_acc": ra, "val_seg_acc": sa}, CHECKPOINT)
            flag = "  *best*"
        history.append(dict(epoch=ep, train_loss=trl, train_acc=tra, val_ref_acc=ra, val_seg_acc=sa,
                            seconds=time.time() - t0))
        print(f"finetune {ep}/{offset + a.epochs}  train {trl:.4f}/{tra:.4f}  val reference {ra:.4f}  "
              f"val segmenter {sa:.4f}  {time.time() - t0:.0f}s{flag}", flush=True)
        hist_path.write_text(json.dumps(history, indent=1))
    print("Done. Best combined validation score", round(best, 4))


if __name__ == "__main__":
    main()

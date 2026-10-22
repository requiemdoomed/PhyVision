"""Train PhyNet on the PhyVision dataset.

    python -m phyvision.train --epochs 16            (run from the src/ folder)
    python scripts/train.py --epochs 16              (from the project root)

AI assistance: this training pipeline was developed with help from
Claude, an AI model by Anthropic, for educational purposes (learning how to
train a vision model).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import CHECKPOINT, CLASSES, REPORTS_DIR, MODELS_DIR, VERSION
from .dataset import SilhouetteDataset
from .model import PhyNet


FRAMING_WEIGHT = 2.0


def run_epoch(model, loader, crit, opt=None, sched=None, stats=None):
    """One pass over `loader`. Loss = cross-entropy + FRAMING_WEIGHT * L1(framing).
    If `stats` is a dict, the mean absolute framing error is stored in it."""
    train = opt is not None
    model.train(train)
    tot, correct, loss_sum, ferr = 0, 0, 0.0, 0.0
    with torch.set_grad_enabled(train):
        for sil, prof, y, vis in loader:
            logits, fr = model(sil, prof)
            loss = crit(logits, y) + FRAMING_WEIGHT * torch.nn.functional.l1_loss(fr, vis)
            ferr += (fr - vis).abs().sum().item()
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            tot += len(y)
    if stats is not None:
        stats["framing_mae"] = ferr / tot
    return loss_sum / tot, correct / tot


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    if a.threads:
        torch.set_num_threads(a.threads)

    print("Loading dataset ...", flush=True)
    tr = SilhouetteDataset("train", augment=True, seed=a.seed)
    va = SilhouetteDataset("val", augment=False)
    print(f"train={len(tr)}  val={len(va)}", flush=True)
    tl = DataLoader(tr, batch_size=a.batch, shuffle=True, num_workers=1, persistent_workers=True,
                    drop_last=True)
    vl = DataLoader(va, batch_size=256, shuffle=False, num_workers=0)

    model = PhyNet()
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(tl),
                                                pct_start=0.15, anneal_strategy="cos")
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    history, best = [], 0.0
    for ep in range(1, a.epochs + 1):
        tr.set_epoch(ep)
        t0 = time.time()
        trl, tra = run_epoch(model, tl, crit, opt, sched)
        st = {}
        val, vaa = run_epoch(model, vl, crit, stats=st)
        history.append(dict(epoch=ep, train_loss=trl, train_acc=tra, val_loss=val, val_acc=vaa,
                            val_framing_mae=st["framing_mae"],
                            lr=sched.get_last_lr()[0], seconds=time.time() - t0))
        flag = ""
        if vaa >= best:
            best = vaa
            torch.save({"state_dict": model.state_dict(), "classes": CLASSES, "version": VERSION,
                        "epoch": ep, "val_acc": vaa}, CHECKPOINT)
            flag = "  *best*"
        print(f"epoch {ep:2d}/{a.epochs}  train {trl:.4f}/{tra:.4f}  val {val:.4f}/{vaa:.4f}  "
              f"framing MAE {st['framing_mae']:.3f}  {time.time() - t0:.0f}s{flag}", flush=True)
        with open(REPORTS_DIR / "history.json", "w") as f:
            json.dump(history, f, indent=1)
    print(f"Best val acc {best:.4f}. Saved {CHECKPOINT}")


if __name__ == "__main__":
    main()

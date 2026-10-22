"""Average the weights of two or more PhyNet checkpoints ("model soup").

Used for the released v5.1 model: the best checkpoint of fine-tuning round 1
and the best checkpoint of round 2 (round 2 continues from round 1, so their
weights are compatible). Averaging gave the same validation accuracy as
round 2 alone and more stable predictions on a real photo.

    python scripts/average_checkpoints.py round1.pt round2.pt --out models/phyvision_phynet.pt
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phyvision.config import CLASSES, VERSION  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cks = [torch.load(p, map_location="cpu", weights_only=False) for p in a.checkpoints]
    avg = {}
    for k, v in cks[0]["state_dict"].items():
        if v.is_floating_point():
            avg[k] = sum(c["state_dict"][k].float() for c in cks) / len(cks)
            avg[k] = avg[k].to(v.dtype)
        else:  # integer buffers (e.g. BatchNorm batch counters): keep the last one
            avg[k] = cks[-1]["state_dict"][k]
    torch.save({"state_dict": avg, "classes": CLASSES, "version": VERSION,
                "epoch": cks[-1].get("epoch"), "averaged_from": [Path(p).name for p in a.checkpoints]}, a.out)
    print(f"Averaged {len(cks)} checkpoints -> {a.out}")


if __name__ == "__main__":
    main()

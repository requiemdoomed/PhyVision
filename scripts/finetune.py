"""Thin wrapper: python scripts/finetune.py [args]  (see src/phyvision/finetune.py)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phyvision.finetune import main  # noqa: E402

if __name__ == "__main__":
    main()

"""Build models/train_hashes.json (MD5 of every training image) for the integrity check."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phyvision.integrity import build_hash_db  # noqa: E402

if __name__ == "__main__":
    db = build_hash_db()
    print(f"Hashed {len(db)} training images")

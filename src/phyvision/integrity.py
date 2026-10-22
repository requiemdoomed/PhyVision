"""MD5 training-set integrity layer (carried over from v4 and extended).

If a user uploads an image that is byte-identical to a training image, the
prediction is flagged so it is never mistaken for genuine generalisation.
"""
import hashlib
import json
from pathlib import Path

from .config import CLASSES, DATA_DIR, HASH_DB


def md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def build_hash_db(root: Path = DATA_DIR, out: Path = HASH_DB) -> dict:
    db = {}
    for c in CLASSES:
        for p in sorted((root / "train" / c).glob("*.jpg")):
            db[md5_bytes(p.read_bytes())] = c
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(db))
    return db


def load_hash_db(path: Path = HASH_DB) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return {}

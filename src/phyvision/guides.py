"""Fitness guide content (shared with the web UI via web/assets/guides.json).
"""
import json
from functools import lru_cache

from .config import WEB_DIR


@lru_cache(maxsize=1)
def load_guides() -> dict:
    return json.loads((WEB_DIR / "assets" / "guides.json").read_text(encoding="utf-8"))


def guide_for(label: str) -> dict:
    return load_guides()["classes"][label]

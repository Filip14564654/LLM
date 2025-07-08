import yaml
import hashlib
import os
import json

def stable_hash(token: str, vocab_size: int) -> int:
    """Return a stable hashed ID for ``token`` within ``vocab_size``."""
    h = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(h, 16) % vocab_size

def load_vocab_mapping(vocab_file: str) -> dict:
    """Load vocabulary mapping from token to ID."""
    with open(vocab_file, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return vocab

def token_to_id(token: str, vocab_mapping: dict, vocab_size: int) -> int:
    """Convert token to ID using vocabulary mapping, fallback to hash if not found."""
    if token in vocab_mapping:
        return vocab_mapping[token] % vocab_size
    else:
        # Fallback to hash for unknown tokens
        return stable_hash(token, vocab_size)

def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def next_version_path(path: str) -> str:
    """Return ``path`` or ``path`` suffixed with ``_vN`` so it doesn't exist."""
    base, ext = os.path.splitext(path)
    version = 0
    candidate = path
    while os.path.exists(candidate):
        version += 1
        candidate = f"{base}_v{version}{ext}"
    return candidate


def latest_version_path(path: str) -> str | None:
    """Return the most recent existing checkpoint matching ``path`` or ``_vN``."""
    base, ext = os.path.splitext(path)
    version = 0
    candidate = path
    latest = None
    while os.path.exists(candidate):
        latest = candidate
        version += 1
        candidate = f"{base}_v{version}{ext}"
    return latest

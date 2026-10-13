"""Text embedders.

`HashingEmbedder` needs only numpy and runs anywhere, which keeps the tool
local and the tests fast. `SentenceTransformerEmbedder` gives much better
semantic similarity (synonyms, paraphrase) and is the recommended
choice for real exports; it downloads the model once and then runs offline.
"""

from __future__ import annotations

import hashlib
from typing import Protocol, Sequence

import numpy as np

from .text import content_tokens


class Embedder(Protocol):
    def fit(self, texts: Sequence[str]) -> "Embedder": ...

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class HashingEmbedder:
    """TF-IDF over hashed unigrams and bigrams, L2-normalized."""

    def __init__(self, dim: int = 4096, bigrams: bool = True):
        self.dim = dim
        self.bigrams = bigrams
        self.idf = np.ones(dim, dtype=np.float32)

    def _features(self, text: str) -> list[int]:
        toks = content_tokens(text)
        feats = toks + ([f"{a} {b}" for a, b in zip(toks, toks[1:])] if self.bigrams else [])
        return [int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little") % self.dim for f in feats]

    def fit(self, texts: Sequence[str]) -> "HashingEmbedder":
        df = np.zeros(self.dim, dtype=np.float32)
        for t in texts:
            for h in set(self._features(t)):
                df[h] += 1
        n = max(len(texts), 1)
        self.idf = (np.log((1 + n) / (1 + df)) + 1).astype(np.float32)
        return self

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        m = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for h in self._features(t):
                m[i, h] += 1
        m = np.log1p(m) * self.idf
        return _normalize(m)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def fit(self, texts: Sequence[str]) -> "SentenceTransformerEmbedder":
        return self

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        return _normalize(np.asarray(self.model.encode(list(texts), show_progress_bar=False), dtype=np.float32))


def get_embedder(name: str = "hashing") -> Embedder:
    if name == "hashing":
        return HashingEmbedder()
    if name.startswith("st:"):
        return SentenceTransformerEmbedder(name[3:])
    if name == "st":
        return SentenceTransformerEmbedder()
    raise ValueError(f"unknown embedder {name!r} (use 'hashing', 'st' or 'st:<model>')")

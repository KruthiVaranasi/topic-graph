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


class LSAEmbedder:
    """Latent semantic analysis on top of the hashing TF-IDF features (numpy only).

    Learns which terms co-occur across the user's own chats ("visa", "consulate",
    "ds160", "interview"), so two turns about the same subject match even when
    they share no exact words. Still fully local; no model download.
    """

    def __init__(self, dim: int = 4096, k: int = 256):
        self.base = HashingEmbedder(dim=dim)
        self.k = k
        self.components: np.ndarray | None = None

    _cache: dict[tuple, tuple] = {}

    def fit(self, texts: Sequence[str]) -> "LSAEmbedder":
        # Fitting is deterministic, so repeated runs on the same corpus (threshold
        # sweeps) reuse the result instead of recomputing the decomposition.
        key = (self.base.dim, self.k, hash(tuple(texts)))
        if key in LSAEmbedder._cache:
            self.base.idf, self.components = LSAEmbedder._cache[key]
            return self
        self.base.fit(texts)
        x = self.base.embed(texts)
        # Eigenvectors of X^T X are the right singular vectors of X.
        vals, vecs = np.linalg.eigh((x.T @ x).astype(np.float64))
        order = np.argsort(vals)[::-1][: self.k]
        self.components = vecs[:, order].astype(np.float32)
        LSAEmbedder._cache[key] = (self.base.idf, self.components)
        return self

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        x = self.base.embed(texts)
        if self.components is None:
            return x
        return _normalize(x @ self.components)


class SentenceTransformerEmbedder:
    """Sentence-BERT embeddings: semantic similarity that survives paraphrase.

    Runs locally after a one-time model download (no API key needed). Vectors
    are memoized per text, so threshold sweeps encode each message only once.
    """

    ALIASES = {"minilm": "all-MiniLM-L6-v2", "bge-small": "BAAI/bge-small-en-v1.5", "mpnet": "all-mpnet-base-v2"}
    _models: dict[str, object] = {}
    _memo: dict[tuple[str, str], np.ndarray] = {}

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model_name = self.ALIASES.get(model_name, model_name)
        if self.model_name not in self._models:
            self._models[self.model_name] = SentenceTransformer(self.model_name)
        self.model = self._models[self.model_name]

    def fit(self, texts: Sequence[str]) -> "SentenceTransformerEmbedder":
        return self

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        missing = list(dict.fromkeys(t for t in texts if (self.model_name, t) not in self._memo))
        if missing:
            vecs = np.asarray(self.model.encode(missing, batch_size=64, show_progress_bar=False), dtype=np.float32)
            for t, v in zip(missing, _normalize(vecs)):
                self._memo[(self.model_name, t)] = v
        return np.stack([self._memo[(self.model_name, t)] for t in texts])


# Thresholds tuned per embedder on the even-numbered chats of evalsets/llm-v1 (`eval --sweep --split tune`).
# Dense embedders score every pair higher, so one set of thresholds cannot serve all of them.
PRESETS = {
    "hashing": {"low": 0.02, "mid": 0.08, "high": 0.26, "merge": None, "cluster_threshold": 0.08},
    "lsa": {"low": 0.08, "mid": 0.17, "high": 0.44, "merge": 0.3, "cluster_threshold": 0.35},
    "st": {"low": 0.15, "mid": 0.28, "high": 0.67, "merge": None, "cluster_threshold": 0.35},
}


def preset_for(name: str) -> dict:
    return PRESETS["st" if name.startswith("st") else "lsa" if name.startswith("lsa") else "hashing"]


def resolve_embedder_name(name: str) -> str:
    """'auto' means Sentence-BERT when it is installed (it scored best), otherwise the numpy-only hashing embedder."""
    if name != "auto":
        return name
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return "hashing"
    return "st:minilm"


def get_embedder(name: str = "hashing") -> Embedder:
    if name == "hashing":
        return HashingEmbedder()
    if name == "lsa":
        return LSAEmbedder()
    if name.startswith("lsa:"):
        return LSAEmbedder(k=int(name[4:]))
    if name.startswith("st:"):
        return SentenceTransformerEmbedder(name[3:])
    if name == "st":
        return SentenceTransformerEmbedder()
    raise ValueError(f"unknown embedder {name!r} (use 'hashing', 'lsa', 'lsa:<k>', 'st' or 'st:<model>')")

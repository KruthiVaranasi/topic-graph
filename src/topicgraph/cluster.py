"""Average-linkage agglomerative clustering on cosine similarity (numpy only).

The number of topics is not known in advance, so clustering stops when the
best remaining pair of clusters is less similar than `threshold`.
O(n^2) memory and roughly O(n^2) per merge: fine up to a few thousand segments.
"""

from __future__ import annotations

from collections import Counter

import numpy as np


def agglomerative(vecs: np.ndarray, threshold: float) -> list[int]:
    n = len(vecs)
    if n == 0:
        return []
    sim = (vecs @ vecs.T).astype(np.float64)
    np.fill_diagonal(sim, -np.inf)
    size = np.ones(n)
    active = np.ones(n, dtype=bool)
    parent = list(range(n))

    while active.sum() > 1:
        flat = int(np.argmax(sim))
        i, j = divmod(flat, n)
        if sim[i, j] < threshold:
            break
        # Merge j into i with the Lance-Williams update for average linkage.
        new = (size[i] * sim[i] + size[j] * sim[j]) / (size[i] + size[j])
        sim[i, :] = new
        sim[:, i] = new
        sim[i, i] = -np.inf
        sim[j, :] = -np.inf
        sim[:, j] = -np.inf
        size[i] += size[j]
        active[j] = False
        parent[j] = i

    def root(k: int) -> int:
        while parent[k] != k:
            k = parent[k]
        return k

    roots = [root(k) for k in range(n)]
    # Renumber clusters 0..k-1, largest first.
    counts = Counter(roots)
    order = sorted(counts, key=lambda r: (-counts[r], r))
    remap = {r: idx for idx, r in enumerate(order)}
    return [remap[r] for r in roots]

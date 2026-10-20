"""Task-level evaluation: can a user find every chat where they discussed a topic?

The segment metrics (topic recall, boundary F1) score the pipeline. This scores the job the user
actually has, "find the chat where I talked about X", and compares Topic Graph against what people
already have: chat titles, and keyword search over full chat text (a generous stand-in for each
app's own search, since it searches all three apps at once).

For each topic there are two queries (evalsets/*/queries.json): a short keyword query and a
"remembered" paraphrase that avoids the topic's own name words. A chat is relevant to a topic when
any of its true segments is about that topic; it is a *hidden* match when that topic is not the
chat's opening segment, so its title says something else.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ..embed import HashingEmbedder, get_embedder
from ..importers import load_exports
from ..pipeline import PipelineConfig, build_graph, search, segment_vectors
from ..segment import SegmenterConfig
from ..text import content_tokens

SBERT_TUNED = PipelineConfig(SegmenterConfig(low=0.15, mid=0.28, high=0.67), cluster_threshold=0.35)


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.toks = [content_tokens(d) for d in docs]
        self.tf = [Counter(t) for t in self.toks]
        self.len = np.array([len(t) for t in self.toks], dtype=float)
        self.avg = self.len.mean() if len(self.len) else 1.0
        df = Counter(w for t in self.toks for w in set(t))
        n = len(docs)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query: str) -> np.ndarray:
        q = content_tokens(query)
        out = np.zeros(len(self.tf))
        for i, tf in enumerate(self.tf):
            norm = self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
            out[i] = sum(self.idf.get(w, 0) * tf[w] * (self.k1 + 1) / (tf[w] + norm) for w in q if w in tf)
        return out

    def has_all(self, query: str) -> np.ndarray:
        q = set(content_tokens(query))
        return np.array([q <= set(t) for t in self.toks])


def _ranked_chats(keys: list[str], scores: np.ndarray, mask: np.ndarray | None = None) -> list[str]:
    """Chat keys by best score, one entry per chat, dropping zero scores (no match)."""
    seen, out = set(), []
    for i in np.argsort(-scores, kind="stable"):
        if scores[i] <= 0 or (mask is not None and not mask[i]):
            continue
        if keys[i] not in seen:
            seen.add(keys[i])
            out.append(keys[i])
    return out


def evaluate_search(data_dir: str | Path, split: str = "test", embedder: str = "st:minilm", ks=(10, 20)) -> dict:
    data_dir = Path(data_dir)
    truth = json.loads((data_dir / "truth.json").read_text(encoding="utf-8"))
    if split != "all":
        want = 0 if split == "tune" else 1
        truth = {k: v for k, v in truth.items() if int(v["synth_id"].split("-")[1]) % 2 == want}
    queries = json.loads((data_dir / "queries.json").read_text(encoding="utf-8"))["queries"]
    chats = [c for c in load_exports([data_dir / "exports"]) if c.key in truth]
    keys = [c.key for c in chats]
    turns = {c.key: c.turns() for c in chats}

    relevant, hidden = defaultdict(set), defaultdict(set)
    for key, t in truth.items():
        for i, seg in enumerate(t["segments"]):
            relevant[seg["topic"]].add(key)
            if i > 0 and seg["topic"] != t["segments"][0]["topic"]:
                hidden[seg["topic"]].add(key)

    # Baselines without a topic graph.
    titles = BM25([c.title for c in chats])
    full = BM25([" ".join(f"{t.user} {t.assistant}" for t in turns[c.key]) for c in chats])
    emb = get_embedder(embedder)
    msg_keys = [k for k in keys for _ in turns[k]]
    msg_vecs = emb.embed([f"{t.user} {t.assistant}" for k in keys for t in turns[k]])

    # Topic Graph as shipped (default embedder and thresholds) and with the tuned Sentence-BERT setting.
    shipped = build_graph(chats, HashingEmbedder(), PipelineConfig()).to_dict()
    tuned_graph = build_graph(chats, get_embedder(embedder), SBERT_TUNED)
    tuned = tuned_graph.to_dict()
    seg_keys = [s.chat_key for s in tuned_graph.segments]
    seg_vecs = segment_vectors(tuned_graph.segments, turns, emb, SBERT_TUNED.segmenter.user_weight)
    node_ids = sorted({s.node_id for s in tuned_graph.segments})
    node_chats = {n: {s.chat_key for s in tuned_graph.segments if s.node_id == n} for n in node_ids}
    node_vecs = np.stack([seg_vecs[[i for i, s in enumerate(tuned_graph.segments) if s.node_id == n]].mean(0) for n in node_ids])
    node_vecs /= np.maximum(np.linalg.norm(node_vecs, axis=1, keepdims=True), 1e-9)

    def methods(q: str) -> dict[str, list[str] | set[str]]:
        qv = emb.embed([q])[0]
        node_order = [node_ids[i] for i in np.argsort(-(node_vecs @ qv))]
        return {
            "Chat titles only (sidebar)": _ranked_chats(keys, titles.scores(q)),
            "Keyword search, all words must match": _ranked_chats(keys, full.scores(q), full.has_all(q)),
            "Keyword search over full chats (BM25)": _ranked_chats(keys, full.scores(q)),
            "Meaning search over messages (no topic graph)": _ranked_chats(msg_keys, msg_vecs @ qv + 1),
            "Topic Graph search, as shipped (default settings)": [h["chat_key"] for h in search(shipped, q, k=10_000)],
            "Topic Graph search, Sentence-BERT": _ranked_chats(seg_keys, seg_vecs @ qv + 1),
            "Topic Graph browse: open the best topic": node_chats[node_order[0]],
            "Topic Graph browse: open the top 3 topics": set().union(*(node_chats[n] for n in node_order[:3])),
        }

    rows = defaultdict(lambda: defaultdict(list))
    for topic, qs in queries.items():
        rel, hid = relevant[topic], hidden[topic]
        if not rel:
            continue
        for kind, q in qs.items():
            for name, result in methods(q).items():
                r = rows[name]
                if isinstance(result, set):  # browsing shows an unranked list
                    r[f"{kind}:recall"].append(len(result & rel) / len(rel))
                    if hid:
                        r[f"{kind}:hidden_recall"].append(len(result & hid) / len(hid))
                    r[f"{kind}:shown"].append(len(result))
                    r[f"{kind}:precision"].append(len(result & rel) / max(len(result), 1))
                else:
                    for k in ks:
                        top = set(dict.fromkeys(result[:k]))
                        r[f"{kind}:recall@{k}"].append(len(top & rel) / len(rel))
                        if hid:
                            r[f"{kind}:hidden_recall@{k}"].append(len(top & hid) / len(hid))
    summary = {name: {m: round(float(np.mean(v)), 4) for m, v in r.items()} for name, r in rows.items()}
    return {
        "chats": len(chats), "topics": len(queries), "queries": 2 * len(queries), "split": split,
        "avg_relevant_chats_per_topic": round(float(np.mean([len(relevant[t]) for t in queries])), 1),
        "avg_hidden_chats_per_topic": round(float(np.mean([len(hidden[t]) for t in queries])), 1),
        "tuned_nodes": len(node_ids), "shipped_nodes": len(shipped["nodes"]),
        "methods": summary,
    }

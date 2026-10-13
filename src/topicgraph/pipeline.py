"""End-to-end: chats -> segments -> labels -> topic nodes -> graph."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .cluster import agglomerative
from .embed import Embedder, HashingEmbedder
from .label import KeywordLabeler, LLMLabeler
from .llm import LLM
from .models import Chat, Edge, Segment, TopicNode, Turn
from .segment import LLMBoundaryJudge, SegmenterConfig, segment_chat


@dataclass
class PipelineConfig:
    segmenter: SegmenterConfig = field(default_factory=SegmenterConfig)
    cluster_threshold: float = 0.08
    min_edge_weight: int = 1


@dataclass
class TopicGraph:
    segments: list[Segment]
    nodes: list[TopicNode]
    edges: list[Edge]

    @property
    def hidden_topic_rate(self) -> float:
        """Share of topic nodes that never appear as the opening (title) segment of a chat."""
        if not self.nodes:
            return 0.0
        return sum(n.hidden_share == 1.0 for n in self.nodes) / len(self.nodes)

    @property
    def hidden_segment_share(self) -> float:
        if not self.segments:
            return 0.0
        return sum(s.hidden for s in self.segments) / len(self.segments)

    def summary(self) -> dict[str, Any]:
        return {
            "chats": len({s.chat_key for s in self.segments}),
            "segments": len(self.segments),
            "topics": len(self.nodes),
            "edges": len(self.edges),
            "hidden_topic_rate": round(self.hidden_topic_rate, 4),
            "hidden_segment_share": round(self.hidden_segment_share, 4),
            "sources": dict(Counter(s.source for s in self.segments)),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "nodes": [asdict(n) for n in self.nodes],
            "edges": [asdict(e) for e in self.edges],
            "segments": [s.to_dict() for s in self.segments],
        }

    def save(self, out_dir: str | Path) -> Path:
        from .viewer import render_html

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        (out / "graph.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        (out / "index.html").write_text(render_html(data), encoding="utf-8")
        return out


def segment_vectors(
    segments: list[Segment], turns_by_chat: dict[str, list[Turn]], embedder: Embedder, user_weight: float = 2.0
) -> np.ndarray:
    if not segments:
        return np.zeros((0, 1), dtype=np.float32)
    spans = [turns_by_chat[s.chat_key][s.start : s.end] for s in segments]
    users = embedder.embed([" ".join(t.user for t in span) for span in spans])
    assistants = embedder.embed([" ".join(t.assistant for t in span) for span in spans])
    v = user_weight * users + assistants
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return v / n


def build_graph(
    chats: list[Chat],
    embedder: Embedder | None = None,
    config: PipelineConfig | None = None,
    llm: LLM | None = None,
    use_llm_judge: bool = True,
    use_llm_labels: bool = True,
    progress: Callable[[str], None] | None = None,
) -> TopicGraph:
    cfg = config or PipelineConfig()
    embedder = embedder or HashingEmbedder()
    say = progress or (lambda _msg: None)

    say(f"fitting embedder on {len(chats)} chats")
    turns_by_chat = {c.key: c.turns() for c in chats}
    all_turns = [t for turns in turns_by_chat.values() for t in turns]
    embedder.fit([t.user for t in all_turns] + [t.assistant for t in all_turns])

    judge = LLMBoundaryJudge(llm) if (llm and use_llm_judge) else None
    segments: list[Segment] = []
    for i, chat in enumerate(chats):
        segments.extend(segment_chat(chat, embedder, cfg.segmenter, judge))
        if (i + 1) % 100 == 0:
            say(f"segmented {i + 1}/{len(chats)} chats")
    say(f"{len(segments)} segments")

    labeler = LLMLabeler(llm) if (llm and use_llm_labels) else KeywordLabeler()
    texts = [s.text for s in segments]
    labeler.fit(texts)
    for s, label in zip(segments, labeler.label_segments(texts)):
        s.label = label

    vecs = segment_vectors(segments, turns_by_chat, embedder, cfg.segmenter.user_weight)
    if isinstance(labeler, LLMLabeler) and segments:
        # LLM labels are a strong, compact topic signal; blend them in.
        lv = embedder.embed([s.label for s in segments])
        vecs = vecs + lv
        vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)
    assignment = agglomerative(vecs, cfg.cluster_threshold)
    for s, a in zip(segments, assignment):
        s.node_id = a
    say(f"{len(set(assignment))} topic nodes")

    members: dict[int, list[Segment]] = defaultdict(list)
    for s in segments:
        members[s.node_id].append(s)
    nodes = []
    for nid in sorted(members):
        segs = members[nid]
        months = Counter(s.timestamp.strftime("%Y-%m") for s in segs if s.timestamp)
        dates = sorted(s.timestamp.isoformat() for s in segs if s.timestamp)
        nodes.append(
            TopicNode(
                id=nid,
                label=labeler.label_node([s.text for s in segs], [s.label for s in segs]),
                segment_ids=[s.id for s in segs],
                sources=dict(Counter(s.source for s in segs)),
                hidden_share=sum(s.hidden for s in segs) / len(segs),
                first_seen=dates[0] if dates else None,
                last_seen=dates[-1] if dates else None,
                by_month=dict(sorted(months.items())),
            )
        )

    return TopicGraph(segments, nodes, build_edges(segments, cfg.min_edge_weight))


def build_edges(segments: list[Segment], min_weight: int = 1) -> list[Edge]:
    by_chat: dict[str, list[Segment]] = defaultdict(list)
    for s in segments:
        by_chat[s.chat_key].append(s)
    cooccur: Counter[tuple[int, int]] = Counter()
    follows: Counter[tuple[int, int]] = Counter()
    for segs in by_chat.values():
        segs.sort(key=lambda s: s.start)
        for a, b in combinations(sorted({s.node_id for s in segs}), 2):
            cooccur[(a, b)] += 1
        for x, y in zip(segs, segs[1:]):
            if x.node_id != y.node_id:
                follows[(x.node_id, y.node_id)] += 1
    edges = [Edge(a, b, "cooccur", w) for (a, b), w in cooccur.items() if w >= min_weight]
    edges += [Edge(a, b, "follows", w) for (a, b), w in follows.items() if w >= min_weight]
    return sorted(edges, key=lambda e: (-e.weight, e.kind, e.source, e.target))


def search(graph_json: dict[str, Any], query: str, embedder: Embedder | None = None, k: int = 10) -> list[dict[str, Any]]:
    """Rank saved segments against a query: "have I solved this before?"."""
    embedder = embedder or HashingEmbedder()
    segs = graph_json["segments"]
    embedder.fit([s["text"] for s in segs])
    m = embedder.embed([s["text"] for s in segs])
    q = embedder.embed([query])[0]
    scores = m @ q
    order = np.argsort(-scores)[:k]
    return [dict(segs[i], score=float(scores[i])) for i in order if scores[i] > 0]

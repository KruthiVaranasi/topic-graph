"""Scoring a topic graph against synthetic ground truth.

Definitions (all micro-averaged unless named macro):

* Boundary F1: predicted topic-shift points vs true ones, a prediction within
  `tolerance` turns of an unmatched true boundary counts as a hit.
* Topic recall: a true segment on topic T is recovered when at least half of
  its turns sit under T's node. T's node is the single node holding the most
  turns of T, so a topic split across two nodes loses recall, as it would for a
  user who opens one node and misses the other.
* Topic precision: a predicted segment is correct when its majority true topic
  equals its node's majority topic.
* Hidden recall: topic recall restricted to true segments that are not the
  chat's first segment, i.e. the ones a title-based organizer cannot find.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from ..models import Segment


def match_boundaries(pred: list[int], true: list[int], tolerance: int = 1) -> int:
    used = set()
    hits = 0
    for p in sorted(pred):
        best = None
        for j, t in enumerate(true):
            if j not in used and abs(p - t) <= tolerance and (best is None or abs(p - t) < abs(p - true[best])):
                best = j
        if best is not None:
            used.add(best)
            hits += 1
    return hits


def _f1(hits: int, n_pred: int, n_true: int) -> dict[str, float]:
    p = hits / n_pred if n_pred else 1.0
    r = hits / n_true if n_true else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "n_pred": n_pred, "n_true": n_true}


def score(segments: list[Segment], truth: dict[str, dict], tolerance: int = 1) -> dict[str, Any]:
    pred_by_chat: dict[str, list[Segment]] = defaultdict(list)
    for s in segments:
        pred_by_chat[s.chat_key].append(s)

    missing = [k for k in truth if k not in pred_by_chat]
    turn_mismatch = [k for k, t in truth.items() if k in pred_by_chat and max(s.end for s in pred_by_chat[k]) != t["n_turns"]]

    # Turn-level labels.
    node_topic_turns: Counter[tuple[int, str]] = Counter()
    turn_node: dict[str, list[int | None]] = {}
    turn_topic: dict[str, list[str]] = {}
    for key, t in truth.items():
        topics = [""] * t["n_turns"]
        for seg in t["segments"]:
            for i in range(seg["start"], seg["end"]):
                topics[i] = seg["topic"]
        nodes: list[int | None] = [None] * t["n_turns"]
        for s in pred_by_chat.get(key, []):
            for i in range(s.start, min(s.end, t["n_turns"])):
                nodes[i] = s.node_id
        turn_topic[key], turn_node[key] = topics, nodes
        for n, tp in zip(nodes, topics):
            if n is not None:
                node_topic_turns[(n, tp)] += 1

    node_major: dict[int, str] = {}
    topic_node: dict[str, int] = {}
    for (n, tp), c in sorted(node_topic_turns.items(), key=lambda x: -x[1]):
        node_major.setdefault(n, tp)
        topic_node.setdefault(tp, n)

    # Boundaries and recall, overall and per case.
    by_case: dict[str, Counter] = defaultdict(Counter)
    recall_by_topic: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for key, t in truth.items():
        case = t["case"]
        c = by_case[case]
        true_b = [s["start"] for s in t["segments"][1:]]
        pred_b = sorted(s.start for s in pred_by_chat.get(key, []) if s.start > 0)
        c["b_hits"] += match_boundaries(pred_b, true_b, tolerance)
        c["b_pred"] += len(pred_b)
        c["b_true"] += len(true_b)
        for seg in t["segments"]:
            span = range(seg["start"], seg["end"])
            target = topic_node.get(seg["topic"])
            hit = sum(turn_node[key][i] == target for i in span) * 2 >= len(span)
            c["r_hit"] += hit
            c["r_n"] += 1
            if seg["start"] > 0:
                c["h_hit"] += hit
                c["h_n"] += 1
            if seg["end"] - seg["start"] == 1:
                c["tiny_hit"] += hit
                c["tiny_n"] += 1
            recall_by_topic[seg["topic"]][0] += hit
            recall_by_topic[seg["topic"]][1] += 1

    # Precision over predicted segments.
    p_hit = p_n = 0
    for key, segs in pred_by_chat.items():
        if key not in truth:
            continue
        for s in segs:
            major = Counter(turn_topic[key][s.start : s.end]).most_common(1)[0][0]
            p_hit += major == node_major.get(s.node_id)
            p_n += 1

    total = sum(by_case.values(), Counter())
    ratio = lambda a, b: round(a / b, 4) if b else None  # noqa: E731
    per_case = {
        case: {
            "chats": sum(1 for t in truth.values() if t["case"] == case),
            "boundary": _f1(c["b_hits"], c["b_pred"], c["b_true"]),
            "topic_recall": ratio(c["r_hit"], c["r_n"]),
            "one_turn_recall": ratio(c["tiny_hit"], c["tiny_n"]),
        }
        for case, c in sorted(by_case.items())
    }
    true_topics = {s["topic"] for t in truth.values() for s in t["segments"]}
    title_topics = {t["segments"][0]["topic"] for t in truth.values()}
    return {
        "chats": len(truth),
        "import": {"missing_chats": len(missing), "turn_count_mismatches": len(turn_mismatch)},
        "boundary": _f1(total["b_hits"], total["b_pred"], total["b_true"]),
        "topic_recall": ratio(total["r_hit"], total["r_n"]),
        "topic_recall_macro": ratio(sum(h / n for h, n in recall_by_topic.values()), len(recall_by_topic)),
        "topic_precision": ratio(p_hit, p_n),
        "hidden_recall": ratio(total["h_hit"], total["h_n"]),
        "one_turn_recall": ratio(total["tiny_hit"], total["tiny_n"]),
        "true_topics": len(true_topics),
        "pred_nodes": len({s.node_id for s in segments}),
        "truth_hidden_topic_rate": ratio(len(true_topics - title_topics), len(true_topics)),
        "truth_hidden_segment_share": ratio(total["h_n"], total["r_n"]),
        "per_case": per_case,
    }


TARGETS = {"topic_recall": 0.90, "topic_precision": 0.80, "boundary_f1": 0.75}


def check_targets(m: dict[str, Any]) -> dict[str, bool]:
    return {
        "topic_recall": (m["topic_recall"] or 0) >= TARGETS["topic_recall"],
        "topic_precision": (m["topic_precision"] or 0) >= TARGETS["topic_precision"],
        "boundary_f1": m["boundary"]["f1"] >= TARGETS["boundary_f1"],
    }

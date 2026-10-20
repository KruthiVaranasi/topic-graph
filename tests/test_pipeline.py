import json
import random

import numpy as np

from topicgraph.cluster import agglomerative
from topicgraph.evaluation.metrics import match_boundaries, score
from topicgraph.models import Chat, Message, Segment
from topicgraph.pipeline import build_graph, build_edges, search
from topicgraph.synth.export import export
from topicgraph.synth.generate import TemplateGenerator
from topicgraph.synth.stitch import DEFAULT_MIX, build_eval_set, plan_chat
from topicgraph.importers import load_exports


def test_agglomerative_separates_clear_groups():
    rng = np.random.default_rng(0)
    centers = np.eye(3)
    vecs = np.vstack([c + 0.05 * rng.standard_normal((5, 3)) for c in centers])
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    labels = agglomerative(vecs, threshold=0.5)
    assert len(set(labels)) == 3
    assert all(len(set(labels[i * 5 : i * 5 + 5])) == 1 for i in range(3))


def test_match_boundaries_tolerance():
    assert match_boundaries([3, 7], [4, 9], tolerance=1) == 1
    assert match_boundaries([4, 5], [4], tolerance=1) == 1  # one true boundary matches once


def test_plan_cases():
    rng = random.Random(0)
    aside = plan_chat("aside", rng)
    assert aside[1][1] == 1 and aside[0][0].id == aside[2][0].id
    ret = plan_chat("return", rng)
    assert ret[0][0].id == ret[-1][0].id
    assert len(plan_chat("single", rng)) == 1


def test_mix_shares():
    chats = build_eval_set(100, TemplateGenerator(), seed=3)
    counts = {c: sum(x.case == c for x in chats) for c in DEFAULT_MIX}
    assert counts == {c: round(100 * s) for c, s in DEFAULT_MIX.items()}
    # Title comes from the first segment.
    assert all(x.title.split()[0] in x.turns[0][0] for x in chats)


def test_drifted_topic_is_findable():
    msgs = [
        Message("user", "python traceback keyerror in my function"),
        Message("assistant", "the keyerror traceback means the dict key is missing in that function"),
        Message("user", "fixed that, now the python traceback shows a nonetype attribute exception in my function"),
        Message("assistant", "that traceback means a variable in the function is none; step through it in the debugger"),
        Message("user", "unrelated, but what documents does the schengen visa embassy appointment need?"),
        Message("assistant", "for a schengen visa appointment bring insurance, itinerary and biometrics documents"),
    ]
    chat = Chat("c1", "Python error", "chatgpt", msgs)
    graph = build_graph([chat])
    assert [(s.start, s.end) for s in graph.segments] == [(0, 2), (2, 3)]
    visa = graph.segments[1]
    assert visa.hidden and "visa" in visa.text
    hits = search(graph.to_dict(), "schengen visa documents")
    assert hits[0]["id"] == visa.id


def test_edges():
    segs = [Segment("a", "t", "chatgpt", 0, 1, "", node_id=0), Segment("a", "t", "chatgpt", 1, 2, "", node_id=1),
            Segment("b", "t", "claude", 0, 1, "", node_id=0), Segment("b", "t", "claude", 1, 2, "", node_id=1)]
    edges = {(e.kind, e.source, e.target): e.weight for e in build_edges(segs)}
    assert edges == {("cooccur", 0, 1): 2, ("follows", 0, 1): 2}


def test_end_to_end_eval_on_template_data(tmp_path):
    chats = build_eval_set(120, TemplateGenerator(), seed=5)
    truth = export(chats, tmp_path, seed=5)
    graph = build_graph(load_exports([tmp_path / "exports"]))
    m = score(graph.segments, truth)
    assert m["import"] == {"missing_chats": 0, "turn_count_mismatches": 0}
    # Template chats are easy; these floors catch regressions, not product claims.
    assert m["topic_recall"] > 0.7
    assert m["topic_precision"] > 0.7
    assert m["boundary"]["f1"] > 0.6
    out = graph.save(tmp_path / "out")
    html = (out / "index.html").read_text(encoding="utf-8")
    assert "<script src" not in html and "<link" not in html  # fully offline viewer
    assert json.loads((out / "graph.json").read_text(encoding="utf-8"))["summary"]["segments"] == len(graph.segments)


def test_search_reuses_saved_vectors_when_given():
    msgs = [Message("user", "python traceback keyerror in my function"),
            Message("assistant", "the keyerror traceback means the dict key is missing"),
            Message("user", "unrelated, but what documents does the schengen visa appointment need?"),
            Message("assistant", "for a schengen visa appointment bring insurance and itinerary documents")]
    graph = build_graph([Chat("c1", "Python error", "chatgpt", msgs)])
    data = graph.to_dict()
    # Vectors that point every segment the "wrong" way prove search uses them instead of re-embedding.
    flipped = graph.vectors[::-1].copy()
    assert search(data, "schengen visa documents", vectors=graph.vectors)[0]["id"] == graph.segments[-1].id
    assert search(data, "schengen visa documents", vectors=flipped)[0]["id"] == graph.segments[0].id


def test_cli_uses_thresholds_tuned_for_the_embedder():
    import argparse

    from topicgraph.cli import _config

    blank = dict(low=None, mid=None, high=None, merge=None, cluster_threshold=None)
    st = _config(argparse.Namespace(embedder="st:minilm", **blank))
    assert (st.segmenter.low, st.segmenter.mid, st.cluster_threshold) == (0.15, 0.28, 0.35)
    hashing = _config(argparse.Namespace(embedder="hashing", **blank))
    assert (hashing.segmenter.mid, hashing.cluster_threshold) == (0.08, 0.08)
    override = _config(argparse.Namespace(embedder="st:minilm", **{**blank, "cluster_threshold": 0.5}))
    assert override.cluster_threshold == 0.5

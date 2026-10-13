import json

from topicgraph.importers import load_exports
from topicgraph.synth import batch


def _fill(out, jobs_glob, write):
    for f in sorted((out / "jobs").glob(jobs_glob)):
        rows = [write(j) for j in json.loads(f.read_text())]
        sub = "segments" if jobs_glob.startswith("segments") else "transitions"
        (out / sub / (f.stem + ".jsonl")).write_text("\n".join(json.dumps(r) for r in rows))


def test_batch_roundtrip(tmp_path):
    counts = batch.plan(tmp_path, n_chats=20, seed=2, batch_size=10)
    assert sum(counts.values()) > 20 and set(counts) == set(batch.TIERS)
    _, problems = batch.validate_segments(tmp_path)
    assert problems and all("missing" in p for p in problems)

    _fill(tmp_path, "segments_*.json", lambda j: {"id": j["id"], "turns": [
        {"user": f"{j['topic']} q{i}", "assistant": f"a{i}"} for i in range(j["n_turns"])]})
    assert batch.validate_segments(tmp_path)[1] == []
    n = batch.transition_jobs(tmp_path)
    assert n == counts["simple"] + counts["medium"] + counts["complex"] - 20
    _fill(tmp_path, "transitions_*.json", lambda j: {"id": j["id"], "message": "btw " + j["message"]})

    chats, truth = batch.assemble(tmp_path)
    loaded = {c.key: c for c in load_exports([tmp_path / "exports"])}
    assert set(loaded) == set(truth)
    for key, t in truth.items():
        turns = loaded[key].turns()
        assert len(turns) == t["n_turns"]
        for seg, tier in zip(t["segments"][1:], t["meta"]["tiers"][1:]):
            assert turns[seg["start"]].user.startswith("btw ") and tier in batch.TIERS


def test_complexity_routing():
    assert batch.complexity("quick fact: x", "polite", 3) == "simple"
    assert batch.complexity("troubleshooting: x", "polite", 2) == "complex"
    assert batch.complexity("how-to: x", "polite", 4) == "medium"
    assert batch.complexity("how-to: x", "polite", 8) == "complex"

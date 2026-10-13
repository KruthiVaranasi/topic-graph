"""Batch generation: plan the eval set, hand segment jobs to external writers, assemble the result.

Used when segments are written outside this process, e.g. by agents or a
batch API, instead of `LLMGenerator` calling a model directly:

1. `plan`:        fix every chat, segment, persona, style, question type and
                  complexity tier from one seed; write job files per tier.
2. (writers fill `segments/*.jsonl` with {"id", "turns": [{"user", "assistant"}]})
3. `transitions`: once segments exist, write the boundary-rewrite jobs.
4. (writers fill `transitions/*.jsonl` with {"id", "message"})
5. `assemble`:    validate everything and export the three tool formats + truth.json.

Each segment gets a complexity tier, so writers can be routed to models of
matching strength: simple to a small model, medium to a mid-size one, and
complex or most human-like segments to the strongest.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .export import export
from .generate import QUESTION_TYPES
from .stitch import DEFAULT_MIX, SyntheticChat, TruthSegment, _plan, _title
from .topics import BY_ID

TIERS = ("simple", "medium", "complex")


def complexity(qtype: str, style: str, n_turns: int) -> str:
    """Route a segment to a model tier by how hard it is to write convincingly.

    simple:  short, generic exchanges (quick facts, one or two turns)
    complex: messy, human-heavy exchanges (debugging, reviewing a draft,
             frustration after a failed attempt) and long back-and-forths
    medium:  everything in between
    """
    if qtype.startswith(("troubleshooting", "feedback")) or (style.startswith("a bit frustrated") and n_turns >= 3) \
            or n_turns >= 7:
        return "complex"
    if qtype.startswith("quick fact") or n_turns <= 2 or (style.startswith("short and direct") and n_turns <= 3):
        return "simple"
    return "medium"


def _read_jsonl(paths) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                row = json.loads(line)
                rows[row["id"]] = row
    return rows


def _chunks(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def plan(out_dir: str | Path, n_chats: int = 300, seed: int = 7, batch_size: int = 25,
         mix: dict[str, float] | None = None) -> dict[str, int]:
    out = Path(out_dir)
    rng = random.Random(seed)
    mix = mix or DEFAULT_MIX
    cases: list[str] = []
    for case, share in mix.items():
        cases += [case] * round(n_chats * share)
    while len(cases) < n_chats:
        cases.append("clean")
    cases = cases[:n_chats]
    rng.shuffle(cases)

    chats, jobs = [], {t: [] for t in TIERS}
    for i, case in enumerate(cases):
        p = _plan(f"synth-{i:04d}", case, rng)
        segs = []
        for k, (topic, n, s) in enumerate(p.segments):
            qtype = random.Random(s).choice(QUESTION_TYPES)
            tier = complexity(qtype, p.style, n)
            job = {"id": f"{p.id}/s{k}", "topic": topic.name, "brief": topic.brief, "n_turns": n,
                   "persona": p.persona, "style": p.style, "question_type": qtype, "tier": tier}
            jobs[tier].append(job)
            segs.append({"id": job["id"], "topic": topic.id, "n_turns": n, "tier": tier, "question_type": qtype})
        chats.append({"id": p.id, "case": case, "persona": p.persona, "style": p.style, "segments": segs})

    (out / "jobs").mkdir(parents=True, exist_ok=True)
    (out / "segments").mkdir(exist_ok=True)
    (out / "plan.json").write_text(json.dumps({"seed": seed, "n_chats": n_chats, "chats": chats}, indent=1), encoding="utf-8")
    for tier, items in jobs.items():
        for b, batch in enumerate(_chunks(items, batch_size)):
            (out / "jobs" / f"segments_{tier}_{b:02d}.json").write_text(json.dumps(batch, indent=1), encoding="utf-8")
    return {t: len(v) for t, v in jobs.items()}


def load_plan(out_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(out_dir) / "plan.json").read_text(encoding="utf-8"))


def validate_segments(out_dir: str | Path) -> tuple[dict[str, dict], list[str]]:
    """Return the generated segments and a list of problems (missing ids, wrong turn counts, blanks)."""
    out = Path(out_dir)
    p = load_plan(out)
    rows = _read_jsonl(sorted((out / "segments").glob("*.jsonl")))
    problems = []
    for chat in p["chats"]:
        for seg in chat["segments"]:
            row = rows.get(seg["id"])
            if row is None:
                problems.append(f"{seg['id']}: missing")
                continue
            turns = row.get("turns") or []
            if len(turns) != seg["n_turns"]:
                problems.append(f"{seg['id']}: {len(turns)} turns, expected {seg['n_turns']}")
            elif any(not str(t.get("user", "")).strip() or not str(t.get("assistant", "")).strip() for t in turns):
                problems.append(f"{seg['id']}: empty message")
    return rows, problems


def transition_jobs(out_dir: str | Path, batch_size: int = 100) -> int:
    out = Path(out_dir)
    rows, problems = validate_segments(out)
    if problems:
        raise ValueError(f"{len(problems)} segment problems, e.g. {problems[:5]}")
    jobs = []
    for chat in load_plan(out)["chats"]:
        segs = chat["segments"]
        for prev, cur in zip(segs, segs[1:]):
            last = rows[prev["id"]]["turns"][-1]
            jobs.append({"id": cur["id"], "previous_user": last["user"], "previous_assistant": last["assistant"][:600],
                         "message": rows[cur["id"]]["turns"][0]["user"]})
    (out / "transitions").mkdir(exist_ok=True)
    for b, batch in enumerate(_chunks(jobs, batch_size)):
        (out / "jobs" / f"transitions_{b:02d}.json").write_text(json.dumps(batch, indent=1, ensure_ascii=False), encoding="utf-8")
    return len(jobs)


def assemble(out_dir: str | Path, export_seed: int | None = None) -> tuple[list[SyntheticChat], dict]:
    out = Path(out_dir)
    p = load_plan(out)
    rows, problems = validate_segments(out)
    if problems:
        raise ValueError(f"{len(problems)} segment problems, e.g. {problems[:5]}")
    trans = _read_jsonl(sorted((out / "transitions").glob("*.jsonl"))) if (out / "transitions").exists() else {}
    chats = []
    for c in p["chats"]:
        turns, segments = [], []
        for k, seg in enumerate(c["segments"]):
            exchanges = [(t["user"].strip(), t["assistant"].strip()) for t in rows[seg["id"]]["turns"]]
            if k > 0:
                if seg["id"] not in trans:
                    raise ValueError(f"missing transition for {seg['id']}")
                exchanges[0] = (trans[seg["id"]]["message"].strip(), exchanges[0][1])
            start = len(turns)
            turns.extend(exchanges)
            segments.append(TruthSegment(start, len(turns), seg["topic"]))
        chat = SyntheticChat(c["id"], c["case"], _title(turns[0][0]), turns, segments, persona=c["persona"], style=c["style"],
                             meta={"tiers": [s["tier"] for s in c["segments"]],
                                   "question_types": [s["question_type"] for s in c["segments"]]})
        chats.append(chat)
    assert all(s.topic in BY_ID for c in chats for s in c.segments)
    truth = export(chats, out, seed=p["seed"] if export_seed is None else export_seed)
    (out / "chats.jsonl").write_text(
        "\n".join(json.dumps({"id": c.id, "tool": c.tool, "case": c.case, "title": c.title, "meta": c.meta,
                              "turns": c.turns}, ensure_ascii=False) for c in chats), encoding="utf-8")
    return chats, truth

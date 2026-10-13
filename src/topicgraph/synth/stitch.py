"""Stitch single-topic segments into drifting chats with known ground truth."""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

from .generate import PERSONAS, STYLES, Exchange, SegmentGenerator, TemplateTransitions, TransitionRewriter
from .topics import BY_ID, TOPICS, Topic, near_pairs

# Share of chats per case, from the eval plan.
DEFAULT_MIX = {"clean": 0.40, "near": 0.20, "return": 0.15, "aside": 0.15, "single": 0.10}


@dataclass
class TruthSegment:
    start: int
    end: int
    topic: str


@dataclass
class SyntheticChat:
    id: str
    case: str
    title: str
    turns: list[Exchange]
    segments: list[TruthSegment]
    tool: str = ""
    persona: str = ""
    style: str = ""
    meta: dict = field(default_factory=dict)

    def truth(self) -> dict:
        return {"case": self.case, "title": self.title, "n_turns": len(self.turns),
                "segments": [asdict(s) for s in self.segments]}


def _title(first_user: str) -> str:
    words = first_user.replace("\n", " ").split()
    return " ".join(words[:6]) + ("…" if len(words) > 6 else "")


def _distinct_domains(rng: random.Random, k: int, exclude: set[str] = frozenset(), avoid: set[str] = frozenset()) -> list[Topic]:
    pool = [t for t in TOPICS if t.id not in exclude and t.domain not in avoid]
    rng.shuffle(pool)
    picked, domains = [], set()
    for t in pool:
        if t.domain not in domains:
            picked.append(t)
            domains.add(t.domain)
        if len(picked) == k:
            break
    return picked


def plan_chat(case: str, rng: random.Random) -> list[tuple[Topic, int]]:
    """Return (topic, n_turns) per segment for one chat of the given case."""
    seg_len = lambda: rng.randint(2, 8)  # noqa: E731
    if case == "single":
        return [(rng.choice(TOPICS), rng.randint(3, 10))]
    if case == "clean":
        return [(t, seg_len()) for t in _distinct_domains(rng, rng.randint(2, 5))]
    if case == "near":
        a, b = (BY_ID[x] for x in rng.choice(near_pairs()))
        if rng.random() < 0.5:
            a, b = b, a
        others = _distinct_domains(rng, rng.randint(0, 2), exclude={a.id, b.id}, avoid={a.domain})
        plan = [(t, seg_len()) for t in others]
        pos = rng.randint(0, len(plan))
        plan[pos:pos] = [(a, seg_len()), (b, seg_len())]
        return plan
    if case == "return":
        ts = _distinct_domains(rng, rng.randint(2, 3))
        a = ts[0]
        return [(a, seg_len())] + [(t, seg_len()) for t in ts[1:]] + [(a, seg_len())]
    if case == "aside":
        a, x = _distinct_domains(rng, 2)
        plan = [(a, rng.randint(2, 6)), (x, 1), (a, rng.randint(1, 5))]
        if rng.random() < 0.3:
            plan.append((_distinct_domains(rng, 1, exclude={a.id, x.id}, avoid={a.domain, x.domain})[0], seg_len()))
        return plan
    raise ValueError(case)


@dataclass
class _ChatPlan:
    id: str
    case: str
    persona: str
    style: str
    segments: list[tuple[Topic, int, int]]  # (topic, n_turns, seed)
    transition_seeds: list[int]


def _plan(chat_id: str, case: str, rng: random.Random) -> _ChatPlan:
    persona, style = rng.choice(PERSONAS), rng.choice(STYLES)
    segs = [(t, n, rng.randrange(2**31)) for t, n in plan_chat(case, rng)]
    return _ChatPlan(chat_id, case, persona, style, segs, [rng.randrange(2**31) for _ in segs[1:]])


def _assemble(plan: _ChatPlan, parts: list[list[Exchange]], transitions: TransitionRewriter, pool) -> SyntheticChat:
    turns: list[Exchange] = []
    segments: list[TruthSegment] = []
    for (topic, _, _), exchanges in zip(plan.segments, parts):
        start = len(turns)
        turns.extend(exchanges)
        segments.append(TruthSegment(start, len(turns), topic.id))
    # Rewrite only the user side of the first turn after each boundary, so it
    # stays on the new topic's side and ground truth remains valid.
    jobs = [
        (seg.start, pool.submit(transitions.rewrite, turns[seg.start - 1], turns[seg.start][0], random.Random(seed)))
        for seg, seed in zip(segments[1:], plan.transition_seeds)
    ]
    for idx, fut in jobs:
        turns[idx] = (fut.result(), turns[idx][1])
    return SyntheticChat(plan.id, plan.case, _title(turns[0][0]), turns, segments, persona=plan.persona, style=plan.style)


def build_eval_set(
    n_chats: int,
    generator: SegmentGenerator,
    seed: int = 7,
    mix: dict[str, float] | None = None,
    transitions: TransitionRewriter | None = None,
    progress=None,
    workers: int = 1,
) -> list[SyntheticChat]:
    """Plan every chat from one seeded RNG, then generate segments (in parallel when workers > 1).

    Each segment gets its own seed at planning time, so the output does not
    depend on the order in which parallel calls finish.
    """
    rng = random.Random(seed)
    mix = mix or DEFAULT_MIX
    transitions = transitions or TemplateTransitions()
    cases: list[str] = []
    for case, share in mix.items():
        cases += [case] * round(n_chats * share)
    while len(cases) < n_chats:
        cases.append("clean")
    cases = cases[:n_chats]
    rng.shuffle(cases)
    plans = [_plan(f"synth-{i:04d}", case, rng) for i, case in enumerate(cases)]

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [
            [pool.submit(generator.generate, t, n, p.persona, p.style, random.Random(s)) for t, n, s in p.segments]
            for p in plans
        ]
        chats = []
        for i, (plan, futs) in enumerate(zip(plans, futures)):
            chats.append(_assemble(plan, [f.result() for f in futs], transitions, pool))
            if progress and (i + 1) % 25 == 0:
                progress(f"generated {i + 1}/{n_chats} chats")
    return chats

"""Split a chat into topic segments.

Each turn is scored by how well it continues the current segment: the higher
of its similarity to the previous turn and to the running segment centroid.
Scores below `low` start a new segment; scores in the borderline band
[`low`, `high`) go to the LLM judge when one is configured, otherwise they are
cut at `mid`. Short follow-ups ("thanks, make it shorter") carry little topical
signal and are kept in the current segment unless the judge says otherwise.
Transition cues ("unrelated, but…") lower the score.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .embed import Embedder
from .llm import LLM, complete_json
from .models import Chat, Segment, Turn
from .text import content_tokens

CUES = re.compile(
    r"^\W*(unrelated|on a different note|different (question|topic)|another (question|thing)|"
    r"switching (gears|topics?)|changing (the )?topic|side (note|question)|btw|by the way|"
    r"also[,:]|quick (question|one)|random (question|q)|new (question|topic))",
    re.I,
)


@dataclass
class SegmenterConfig:
    # Calibrated for HashingEmbedder with `topicgraph eval --sweep`; dense
    # embedders score higher overall, so re-run the sweep when switching.
    low: float = 0.02
    mid: float = 0.08
    high: float = 0.26
    cue_penalty: float = 0.10
    min_tokens: int = 3  # user turns with fewer content tokens count as follow-ups
    user_weight: float = 2.0  # the user message carries more topic signal than the reply


class LLMBoundaryJudge:
    SYSTEM = (
        "You decide whether a new message in an AI-assistant chat continues the current topic "
        "or starts a different one. A follow-up, clarification or deeper question on the same "
        "subject is the same topic. A switch to an unrelated subject, even briefly, is a new topic."
    )
    SCHEMA = {
        "type": "object",
        "properties": {"new_topic": {"type": "boolean"}},
        "required": ["new_topic"],
        "additionalProperties": False,
    }

    def __init__(self, llm: LLM, context_turns: int = 3, max_chars: int = 1200):
        self.llm = llm
        self.context_turns = context_turns
        self.max_chars = max_chars

    def _clip(self, s: str) -> str:
        return s if len(s) <= self.max_chars else s[: self.max_chars] + " […]"

    def is_boundary(self, context: list[Turn], turn: Turn) -> bool:
        ctx = "\n\n".join(
            f"User: {self._clip(t.user)}\nAssistant: {self._clip(t.assistant)}" for t in context[-self.context_turns :]
        )
        prompt = (
            f"<current_topic_turns>\n{ctx}\n</current_topic_turns>\n\n"
            f"<new_turn>\nUser: {self._clip(turn.user)}\nAssistant: {self._clip(turn.assistant)}\n</new_turn>\n\n"
            "Does the new turn start a different topic?"
        )
        return bool(complete_json(self.llm, self.SYSTEM, prompt, self.SCHEMA)["new_topic"])


def turn_vectors(turns: list[Turn], embedder: Embedder, user_weight: float) -> np.ndarray:
    if not turns:
        return np.zeros((0, 1), dtype=np.float32)
    u = embedder.embed([t.user for t in turns])
    a = embedder.embed([t.assistant for t in turns])
    v = user_weight * u + a
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return v / n


def segment_chat(
    chat: Chat,
    embedder: Embedder,
    config: SegmenterConfig | None = None,
    judge: LLMBoundaryJudge | None = None,
) -> list[Segment]:
    cfg = config or SegmenterConfig()
    turns = chat.turns()
    if not turns:
        return []
    vecs = turn_vectors(turns, embedder, cfg.user_weight)
    starts = [0]
    centroid = vecs[0].copy()
    for i in range(1, len(turns)):
        v = vecs[i]
        c = centroid / (np.linalg.norm(centroid) or 1.0)
        score = max(float(v @ vecs[i - 1]), float(v @ c))
        cue = bool(CUES.match(turns[i].user))
        if cue:
            score -= cfg.cue_penalty
        short = len(content_tokens(turns[i].user)) < cfg.min_tokens and not cue

        if short:
            cut = judge.is_boundary(turns[starts[-1] : i], turns[i]) if judge else False
        elif score < cfg.low:
            cut = True
        elif score < cfg.high:
            cut = judge.is_boundary(turns[starts[-1] : i], turns[i]) if judge else score < cfg.mid
        else:
            cut = False

        if cut:
            starts.append(i)
            centroid = v.copy()
        else:
            centroid = centroid + v
    bounds = starts + [len(turns)]
    segments = []
    for s, e in zip(bounds, bounds[1:]):
        seg_turns = turns[s:e]
        segments.append(
            Segment(
                chat_key=chat.key,
                chat_title=chat.title,
                source=chat.source,
                start=s,
                end=e,
                text="\n\n".join(f"User: {t.user}\nAssistant: {t.assistant}" for t in seg_turns),
                timestamp=seg_turns[0].timestamp or chat.created_at,
                url=chat.url,
            )
        )
    return segments

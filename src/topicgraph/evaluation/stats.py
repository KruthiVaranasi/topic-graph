"""Distribution check: compare synthetic chats with real exports on simple shape statistics."""

from __future__ import annotations

import statistics
from typing import Any

from ..models import Chat


def _summary(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {"n": 0}
    xs = sorted(xs)
    return {
        "n": len(xs),
        "mean": round(statistics.fmean(xs), 1),
        "median": xs[len(xs) // 2],
        "p90": xs[min(len(xs) - 1, int(len(xs) * 0.9))],
    }


def chat_stats(chats: list[Chat], segments_per_chat: list[int] | None = None) -> dict[str, Any]:
    turns = [c.turns() for c in chats]
    out = {
        "chats": len(chats),
        "by_source": {s: sum(c.source == s for c in chats) for s in sorted({c.source for c in chats})},
        "turns_per_chat": _summary([len(t) for t in turns]),
        "user_words": _summary([len(x.user.split()) for ts in turns for x in ts]),
        "assistant_words": _summary([len(x.assistant.split()) for ts in turns for x in ts]),
    }
    if segments_per_chat is not None:
        out["topics_per_chat"] = _summary(segments_per_chat)
    return out

"""Short topic names for segments and topic nodes."""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

from .llm import LLM, complete_json
from .text import content_tokens


class KeywordLabeler:
    """Top TF-IDF terms. No LLM needed; labels read like "visa / schengen / appointment"."""

    def __init__(self, n_terms: int = 3):
        self.n_terms = n_terms
        self.df: Counter[str] = Counter()
        self.n_docs = 0

    def fit(self, texts: Sequence[str]) -> "KeywordLabeler":
        self.df = Counter()
        for t in texts:
            self.df.update(set(content_tokens(t)))
        self.n_docs = len(texts)
        return self

    def _top(self, texts: Sequence[str]) -> str:
        tf: Counter[str] = Counter()
        for t in texts:
            tf.update(content_tokens(t))
        # A term must recur (or the text be tiny) to name a topic.
        scored = {
            w: c * math.log((1 + self.n_docs) / (1 + self.df[w]))
            for w, c in tf.items()
            if c > 1 or sum(tf.values()) < 12
        }
        top = sorted(scored, key=lambda w: (-scored[w], w))[: self.n_terms]
        return " / ".join(top) or "misc"

    def label_segments(self, texts: Sequence[str]) -> list[str]:
        return [self._top([t]) for t in texts]

    def label_node(self, segment_texts: Sequence[str], segment_labels: Sequence[str]) -> str:
        return self._top(segment_texts)


class LLMLabeler:
    SEGMENT_SYSTEM = (
        "You name the topic of an excerpt from an AI-assistant chat. Reply with a short, specific "
        "topic name of 2 to 5 words in English, like 'Python import errors' or 'Schengen visa documents'. "
        "Name the subject, not the user's intent."
    )
    NODE_SYSTEM = (
        "You are given topic names that were assigned to related excerpts from AI-assistant chats. "
        "Reply with one short, specific name of 2 to 5 words that covers them."
    )
    SCHEMA = {
        "type": "object",
        "properties": {"topic": {"type": "string"}},
        "required": ["topic"],
        "additionalProperties": False,
    }

    def __init__(self, llm: LLM, max_chars: int = 4000):
        self.llm = llm
        self.max_chars = max_chars

    def fit(self, texts: Sequence[str]) -> "LLMLabeler":
        return self

    def label_segments(self, texts: Sequence[str]) -> list[str]:
        return [
            complete_json(self.llm, self.SEGMENT_SYSTEM, f"<excerpt>\n{t[: self.max_chars]}\n</excerpt>", self.SCHEMA)["topic"]
            for t in texts
        ]

    def label_node(self, segment_texts: Sequence[str], segment_labels: Sequence[str]) -> str:
        counts = Counter(segment_labels)
        if len(counts) == 1:
            return segment_labels[0]
        names = "\n".join(f"- {name} (x{n})" for name, n in counts.most_common(25))
        return complete_json(self.llm, self.NODE_SYSTEM, names, self.SCHEMA)["topic"]

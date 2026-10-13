"""Core data types shared by every stage of the pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Message:
    role: str  # "user" or "assistant"
    text: str
    timestamp: datetime | None = None


@dataclass
class Turn:
    """One exchange: a user message and the assistant reply that follows it.

    Turns are the unit of segmentation: a topic boundary always falls between
    two turns, never inside one.
    """

    index: int
    user: str
    assistant: str
    timestamp: datetime | None = None

    @property
    def text(self) -> str:
        return f"{self.user}\n{self.assistant}".strip()


@dataclass
class Chat:
    id: str
    title: str
    source: str  # "chatgpt", "claude" or "gemini"
    messages: list[Message] = field(default_factory=list)
    created_at: datetime | None = None
    url: str | None = None

    @property
    def key(self) -> str:
        return f"{self.source}:{self.id}"

    def turns(self) -> list[Turn]:
        turns: list[Turn] = []
        user, assistant, ts = None, [], None
        for msg in self.messages:
            if msg.role == "user":
                if user is not None or assistant:
                    turns.append(Turn(len(turns), user or "", "\n".join(assistant), ts))
                user, assistant, ts = msg.text, [], msg.timestamp
            else:
                assistant.append(msg.text)
                ts = ts or msg.timestamp
        if user is not None or assistant:
            turns.append(Turn(len(turns), user or "", "\n".join(assistant), ts))
        return turns


@dataclass
class Segment:
    """A contiguous run of turns inside one chat that stays on one topic."""

    chat_key: str
    chat_title: str
    source: str
    start: int  # first turn index
    end: int  # one past the last turn index
    text: str
    label: str = ""
    timestamp: datetime | None = None
    url: str | None = None
    node_id: int | None = None

    @property
    def id(self) -> str:
        return f"{self.chat_key}#{self.start}-{self.end}"

    @property
    def hidden(self) -> bool:
        """True when the segment is not the one the chat title came from."""
        return self.start > 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["id"] = self.id
        d["hidden"] = self.hidden
        d["timestamp"] = self.timestamp.isoformat() if self.timestamp else None
        return d


@dataclass
class TopicNode:
    id: int
    label: str
    segment_ids: list[str]
    sources: dict[str, int]
    hidden_share: float
    first_seen: str | None
    last_seen: str | None
    by_month: dict[str, int]


@dataclass
class Edge:
    source: int
    target: int
    kind: str  # "cooccur" (same chat) or "follows" (source segment directly precedes target)
    weight: int

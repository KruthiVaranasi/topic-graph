"""Claude data export (Settings > Privacy > Export data): `conversations.json`."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ..models import Chat, Message


def _ts(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _message_text(msg: dict[str, Any]) -> str:
    blocks = msg.get("content")
    if isinstance(blocks, list) and blocks:
        text = "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        if text.strip():
            return text.strip()
    return (msg.get("text") or "").strip()


def parse_conversations(data: list[dict[str, Any]]) -> list[Chat]:
    chats = []
    for conv in data:
        messages = []
        for msg in conv.get("chat_messages") or []:
            role = {"human": "user", "assistant": "assistant"}.get(msg.get("sender"))
            text = _message_text(msg)
            if role and text:
                messages.append(Message(role, text, _ts(msg.get("created_at"))))
        if messages:
            uuid = conv.get("uuid", "")
            chats.append(
                Chat(
                    id=uuid,
                    title=conv.get("name") or "Untitled",
                    source="claude",
                    messages=messages,
                    created_at=_ts(conv.get("created_at")),
                    url=f"https://claude.ai/chat/{uuid}" if uuid else None,
                )
            )
    return chats


def load(path: Path) -> list[Chat]:
    return parse_conversations(json.loads(path.read_text(encoding="utf-8")))


def looks_like(path: Path, head: str) -> bool:
    return '"chat_messages"' in head

"""ChatGPT data export: `conversations.json`, or `chat.html` which embeds the same JSON."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..models import Chat, Message


def _ts(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    return datetime.fromtimestamp(float(value), tz=timezone.utc)


def _message_text(msg: dict[str, Any]) -> str:
    content = msg.get("content") or {}
    if content.get("content_type") not in (None, "text", "multimodal_text"):
        return ""  # code-interpreter output, browsing results, etc.
    parts = content.get("parts") or []
    return "\n".join(p for p in parts if isinstance(p, str)).strip()


def _linear_path(conv: dict[str, Any]) -> list[dict[str, Any]]:
    """Follow parent links from `current_node` back to the root.

    The mapping is a tree because edits and regenerations create branches;
    `current_node` marks the branch the user last saw.
    """
    mapping = conv.get("mapping") or {}
    node_id = conv.get("current_node")
    if node_id not in mapping:  # fall back to the latest leaf
        leaves = [n for n in mapping.values() if not n.get("children") and n.get("message")]
        if not leaves:
            return []
        node_id = max(leaves, key=lambda n: n["message"].get("create_time") or 0)["id"]
    path = []
    seen = set()
    while node_id and node_id in mapping and node_id not in seen:
        seen.add(node_id)
        node = mapping[node_id]
        if node.get("message"):
            path.append(node["message"])
        node_id = node.get("parent")
    return list(reversed(path))


def parse_conversations(data: list[dict[str, Any]]) -> list[Chat]:
    chats = []
    for conv in data:
        cid = conv.get("conversation_id") or conv.get("id") or ""
        messages = []
        for msg in _linear_path(conv):
            role = (msg.get("author") or {}).get("role")
            if role not in ("user", "assistant"):
                continue
            if (msg.get("metadata") or {}).get("is_visually_hidden_from_conversation"):
                continue
            text = _message_text(msg)
            if text:
                messages.append(Message(role, text, _ts(msg.get("create_time"))))
        if messages:
            chats.append(
                Chat(
                    id=cid,
                    title=conv.get("title") or "Untitled",
                    source="chatgpt",
                    messages=messages,
                    created_at=_ts(conv.get("create_time")),
                    url=f"https://chatgpt.com/c/{cid}" if cid else None,
                )
            )
    return chats


def extract_json_from_html(html: str) -> list[dict[str, Any]]:
    marker = "jsonData"
    idx = html.find(marker)
    if idx < 0:
        raise ValueError("chat.html does not contain the embedded jsonData array")
    start = html.index("[", idx)
    data, _ = json.JSONDecoder().raw_decode(html[start:])
    return data


def load(path: Path) -> list[Chat]:
    text = path.read_text(encoding="utf-8")
    data = extract_json_from_html(text) if path.suffix.lower() in (".html", ".htm") else json.loads(text)
    return parse_conversations(data)


def looks_like(path: Path, head: str) -> bool:
    if path.suffix.lower() in (".html", ".htm"):
        return "jsonData" in head
    return '"mapping"' in head

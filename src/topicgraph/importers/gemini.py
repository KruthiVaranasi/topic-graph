"""Gemini via Google Takeout: `My Activity/Gemini Apps/MyActivity.json` (or `.html`).

Takeout stores Gemini as a flat activity log of prompt/response pairs with no
conversation ids, so chats are rebuilt as sessions: consecutive prompts less
than `SESSION_GAP` apart belong to the same chat. The first prompt becomes the
title, mimicking how the Gemini sidebar names chats.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from ..models import Chat, Message
from ..text import strip_html

SESSION_GAP = timedelta(minutes=30)
_PROMPTED = re.compile(r"^\s*Prompted\s+", re.I)


def session_id(first_prompt_time: datetime) -> str:
    """Stable chat id for a rebuilt session. Shared with the synthetic exporter."""
    return "gemini-" + first_prompt_time.strftime("%Y%m%dT%H%M%S")


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _title(prompt: str) -> str:
    words = prompt.split()
    return " ".join(words[:8]) + ("…" if len(words) > 8 else "")


def build_sessions(items: list[tuple[datetime, str, str]]) -> list[Chat]:
    """items: (time, prompt, response) tuples in any order."""
    chats: list[Chat] = []
    last: datetime | None = None
    for when, prompt, response in sorted(items, key=lambda x: x[0]):
        if last is None or when - last > SESSION_GAP:
            sid = session_id(when)
            chats.append(Chat(id=sid, title=_title(prompt), source="gemini", created_at=when, url="https://gemini.google.com/app"))
        chats[-1].messages.append(Message("user", prompt, when))
        if response:
            chats[-1].messages.append(Message("assistant", response, when))
        last = when
    return chats


def parse_activity(data: list[dict[str, Any]]) -> list[Chat]:
    items = []
    for entry in data:
        title = entry.get("title") or ""
        if not _PROMPTED.match(title) or not entry.get("time"):
            continue  # feedback, settings changes, etc.
        prompt = _PROMPTED.sub("", title, count=1).strip()
        response = "\n".join(strip_html(x.get("html", "")) for x in entry.get("safeHtmlItem") or [])
        items.append((_ts(entry["time"]), prompt, response.strip()))
    return build_sessions(items)


def _html_time(text: str) -> datetime | None:
    try:
        return _ts(text)
    except ValueError:
        pass
    # Takeout's display format, e.g. "Oct 1, 2025, 10:00:00 AM IST". The zone
    # abbreviation is dropped; only ordering and gaps matter for sessions.
    text = re.sub(r"\s+[A-Z]{2,5}$", "", text.replace("\u202f", " ").strip())
    for fmt in ("%b %d, %Y, %I:%M:%S %p", "%d %b %Y, %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


class _ActivityHTML(HTMLParser):
    """Collects the text and inner HTML of each `content-cell` in MyActivity.html."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.cells: list[str] = []
        self._depth = 0
        self._buf: list[str] = []

    def handle_starttag(self, tag, attrs):
        if self._depth:
            self._depth += tag == "div"
            self._buf.append(self.get_starttag_text() or "")
        elif tag == "div" and "content-cell" in (dict(attrs).get("class") or ""):
            self._depth, self._buf = 1, []

    def handle_endtag(self, tag):
        if not self._depth:
            return
        if tag == "div":
            self._depth -= 1
            if not self._depth:
                self.cells.append("".join(self._buf))
                return
        self._buf.append(f"</{tag}>")

    def handle_data(self, data):
        if self._depth:
            self._buf.append(data)

    def handle_entityref(self, name):
        if self._depth:
            self._buf.append(f"&{name};")

    def handle_charref(self, name):
        if self._depth:
            self._buf.append(f"&#{name};")


def parse_activity_html(html: str) -> list[Chat]:
    """Best-effort parser. Prefer the JSON export (choose JSON in Takeout's format options)."""
    p = _ActivityHTML()
    p.feed(html)
    items = []
    for cell in p.cells:
        # Layout: "Prompted <prompt><br><time><br><response html>"
        pieces = re.split(r"<br\s*/?>", cell, maxsplit=2)
        if len(pieces) < 2 or not _PROMPTED.match(strip_html(pieces[0])):
            continue
        prompt = _PROMPTED.sub("", strip_html(pieces[0]), count=1).strip()
        when = _html_time(strip_html(pieces[1]))
        if when is None:
            continue
        response = strip_html(pieces[2]) if len(pieces) > 2 else ""
        items.append((when, prompt, response))
    return build_sessions(items)


def load(path: Path) -> list[Chat]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".html", ".htm"):
        return parse_activity_html(text)
    return parse_activity(json.loads(text))


def looks_like(path: Path, head: str) -> bool:
    return "Gemini Apps" in head or ("Prompted" in head and "safeHtmlItem" in head)

"""Import chat exports. Each importer is a thin module with `load(path)` and `looks_like(path, head)`."""

from __future__ import annotations

from pathlib import Path

from ..models import Chat
from . import chatgpt, claude, gemini

IMPORTERS = {"chatgpt": chatgpt, "claude": claude, "gemini": gemini}


def detect(path: Path) -> str | None:
    head = path.read_text(encoding="utf-8", errors="ignore")[:200_000]
    # Gemini first: its HTML has no jsonData; Claude before ChatGPT since both are JSON lists.
    for name in ("gemini", "claude", "chatgpt"):
        if IMPORTERS[name].looks_like(path, head):
            return name
    return None


def _candidates(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in (".json", ".html", ".htm") and p.is_file())


def load_exports(paths: list[str | Path], source: str | None = None) -> list[Chat]:
    """Load every recognizable export under `paths` (files or directories)."""
    chats: dict[str, Chat] = {}
    for raw in paths:
        for path in _candidates(Path(raw)):
            kind = source or detect(path)
            if kind is None:
                continue
            # ChatGPT ships the same chats as JSON and HTML; keep one copy.
            for chat in IMPORTERS[kind].load(path):
                chats.setdefault(chat.key, chat)
    return list(chats.values())

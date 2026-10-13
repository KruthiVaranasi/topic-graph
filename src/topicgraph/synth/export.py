"""Write synthetic chats in each tool's real export format, plus a ground-truth file.

Running the eval through these files tests the importers too.
"""

from __future__ import annotations

import html
import json
import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..importers.gemini import session_id
from .stitch import SyntheticChat

DEFAULT_TOOL_SHARE = {"chatgpt": 0.45, "claude": 0.35, "gemini": 0.20}
BASE_TIME = datetime(2025, 1, 6, 9, 0, tzinfo=timezone.utc)
TURN_GAP = timedelta(minutes=2)


def _uuid(rng: random.Random) -> str:
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


def _iso(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%S.000000Z")


def chatgpt_conversation(chat: SyntheticChat, start: datetime, rng: random.Random) -> dict:
    cid = _uuid(rng)
    root = _uuid(rng)
    mapping = {root: {"id": root, "message": None, "parent": None, "children": []}}
    parent, t = root, start
    for user, assistant in chat.turns:
        for role, text in (("user", user), ("assistant", assistant)):
            mid = _uuid(rng)
            mapping[mid] = {
                "id": mid,
                "message": {
                    "id": mid,
                    "author": {"role": role, "name": None, "metadata": {}},
                    "create_time": t.timestamp(),
                    "content": {"content_type": "text", "parts": [text]},
                    "status": "finished_successfully",
                    "metadata": {},
                },
                "parent": parent,
                "children": [],
            }
            mapping[parent]["children"].append(mid)
            parent = mid
        t += TURN_GAP
    return {
        "title": chat.title,
        "create_time": start.timestamp(),
        "update_time": t.timestamp(),
        "mapping": mapping,
        "current_node": parent,
        "conversation_id": cid,
        "id": cid,
    }


def claude_conversation(chat: SyntheticChat, start: datetime, rng: random.Random) -> dict:
    msgs, t = [], start
    for user, assistant in chat.turns:
        for sender, text in (("human", user), ("assistant", assistant)):
            msgs.append({"uuid": _uuid(rng), "text": text, "content": [{"type": "text", "text": text}],
                         "sender": sender, "created_at": _iso(t), "updated_at": _iso(t), "attachments": [], "files": []})
        t += TURN_GAP
    return {"uuid": _uuid(rng), "name": chat.title, "created_at": _iso(start), "updated_at": _iso(t),
            "account": {"uuid": "synthetic"}, "chat_messages": msgs}


def gemini_activities(chat: SyntheticChat, start: datetime) -> list[dict]:
    out, t = [], start
    for user, assistant in chat.turns:
        body = "".join(f"<p>{html.escape(p)}</p>" for p in assistant.split("\n") if p.strip())
        out.append({"header": "Gemini Apps", "title": f"Prompted {user}", "time": _iso(t),
                    "products": ["Gemini Apps"], "activityControls": ["Gemini Apps Activity"],
                    "safeHtmlItem": [{"html": body}]})
        t += TURN_GAP
    return out


def export(chats: list[SyntheticChat], out_dir: str | Path, seed: int = 7,
           tool_share: dict[str, float] | None = None) -> dict:
    """Write exports under out_dir/exports/<tool>/ and ground truth to out_dir/truth.json."""
    rng = random.Random(seed + 1)
    share = tool_share or DEFAULT_TOOL_SHARE
    out = Path(out_dir)
    tools, weights = zip(*share.items())
    # Chats are spaced 6h apart (longer than any chat), so Gemini sessions never merge.
    slots = list(range(len(chats)))
    rng.shuffle(slots)

    chatgpt, claude, gemini, truth = [], [], [], {}
    for chat, slot in zip(chats, slots):
        chat.tool = rng.choices(tools, weights)[0]
        start = BASE_TIME + timedelta(hours=6 * slot, minutes=rng.randint(0, 60))
        if chat.tool == "chatgpt":
            conv = chatgpt_conversation(chat, start, rng)
            chatgpt.append(conv)
            key = f"chatgpt:{conv['conversation_id']}"
        elif chat.tool == "claude":
            conv = claude_conversation(chat, start, rng)
            claude.append(conv)
            key = f"claude:{conv['uuid']}"
        else:
            gemini.extend(gemini_activities(chat, start))
            key = f"gemini:{session_id(start)}"
        truth[key] = dict(chat.truth(), synth_id=chat.id, tool=chat.tool, persona=chat.persona, style=chat.style)

    def write(path: Path, data) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    write(out / "exports/chatgpt/conversations.json", chatgpt)
    (out / "exports/chatgpt/chat.html").write_text(
        "<html><body><script>var jsonData = " + json.dumps(chatgpt, ensure_ascii=False).replace("</", "<\\/")
        + ";</script></body></html>", encoding="utf-8")
    write(out / "exports/claude/conversations.json", claude)
    gemini.sort(key=lambda a: a["time"], reverse=True)  # Takeout lists newest first
    write(out / "exports/gemini/My Activity/Gemini Apps/MyActivity.json", gemini)
    write(out / "truth.json", truth)
    return truth

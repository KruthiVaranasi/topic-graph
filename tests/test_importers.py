import json

from topicgraph.importers import chatgpt, claude, detect, gemini, load_exports
from topicgraph.synth.export import export
from topicgraph.synth.generate import TemplateGenerator
from topicgraph.synth.stitch import build_eval_set


def _conv_with_branch():
    # root -> u1 -> a1 -> u2 (edited away) ; a1 -> u2b -> a2b (current branch)
    def node(i, role, text, parent, children, t):
        return {"id": i, "parent": parent, "children": children, "message": {
            "id": i, "author": {"role": role}, "create_time": t,
            "content": {"content_type": "text", "parts": [text]}, "metadata": {}}}

    mapping = {
        "root": {"id": "root", "parent": None, "children": ["u1"], "message": None},
        "u1": node("u1", "user", "python keyerror help", "root", ["a1"], 1),
        "a1": node("a1", "assistant", "check the dict keys", "u1", ["u2", "u2b"], 2),
        "u2": node("u2", "user", "old edited question", "a1", [], 3),
        "u2b": node("u2b", "user", "unrelated, schengen visa documents?", "a1", ["a2b"], 4),
        "a2b": node("a2b", "assistant", "you need insurance and an itinerary", "u2b", [], 5),
    }
    return {"title": "Python error", "create_time": 1, "mapping": mapping, "current_node": "a2b", "conversation_id": "c1"}


def test_chatgpt_follows_current_branch(tmp_path):
    chats = chatgpt.parse_conversations([_conv_with_branch()])
    assert len(chats) == 1
    texts = [m.text for m in chats[0].messages]
    assert "old edited question" not in texts
    assert texts[-1] == "you need insurance and an itinerary"
    assert len(chats[0].turns()) == 2
    assert chats[0].url == "https://chatgpt.com/c/c1"


def test_chatgpt_html(tmp_path):
    p = tmp_path / "chat.html"
    p.write_text("<html><script>var jsonData = " + json.dumps([_conv_with_branch()]) + ";</script></html>", encoding="utf-8")
    assert detect(p) == "chatgpt"
    assert chatgpt.load(p)[0].title == "Python error"


def test_claude_export(tmp_path):
    data = [{"uuid": "u", "name": "Resume", "created_at": "2025-01-01T10:00:00Z", "chat_messages": [
        {"sender": "human", "text": "fix my resume", "created_at": "2025-01-01T10:00:00Z"},
        {"sender": "assistant", "text": "", "content": [{"type": "text", "text": "sure"}], "created_at": "2025-01-01T10:00:01Z"},
    ]}]
    p = tmp_path / "conversations.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    assert detect(p) == "claude"
    [chat] = claude.load(p)
    assert [t.assistant for t in chat.turns()] == ["sure"]


def test_gemini_sessions_split_on_gap():
    data = [
        {"header": "Gemini Apps", "title": "Prompted second", "time": "2025-01-01T10:05:00Z", "safeHtmlItem": [{"html": "<p>b</p>"}]},
        {"header": "Gemini Apps", "title": "Prompted first", "time": "2025-01-01T10:00:00Z", "safeHtmlItem": [{"html": "<p>a &amp; c</p>"}]},
        {"header": "Gemini Apps", "title": "Prompted later", "time": "2025-01-01T15:00:00Z"},
        {"header": "Gemini Apps", "title": "Gave feedback", "time": "2025-01-01T15:01:00Z"},
    ]
    chats = gemini.parse_activity(data)
    assert [len(c.turns()) for c in chats] == [2, 1]
    assert chats[0].title == "first"
    assert chats[0].messages[1].text == "a & c"


def test_gemini_html():
    html = (
        '<div class="outer-cell"><div class="content-cell mdl-cell">Prompted&nbsp;how to proof dough<br>'
        "Jan 1, 2025, 10:00:00 AM UTC<br><p>Let it rise <b>twice</b>.</p></div>"
        '<div class="content-cell mdl-cell">Products: Gemini Apps</div></div>'
    )
    [chat] = gemini.parse_activity_html(html)
    assert chat.messages[0].text == "how to proof dough"
    assert "twice" in chat.messages[1].text


def test_synthetic_export_roundtrip(tmp_path):
    chats = build_eval_set(30, TemplateGenerator(), seed=1)
    truth = export(chats, tmp_path, seed=1)
    loaded = {c.key: c for c in load_exports([tmp_path / "exports"])}
    assert set(loaded) == set(truth)
    for key, t in truth.items():
        assert len(loaded[key].turns()) == t["n_turns"]

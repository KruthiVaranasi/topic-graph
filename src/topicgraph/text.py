"""Small text utilities: tokenizing, stopwords and HTML stripping (stdlib only)."""

from __future__ import annotations

import re
from html import unescape
from html.parser import HTMLParser

_TOKEN = re.compile(r"[a-z0-9][a-z0-9+#._-]*[a-z0-9+#]|[a-z0-9]")

# English stopwords plus chat filler and common Hinglish function words, so
# "yaar kya karu" does not look like a topic.
STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are as at be because been before being below
    between both but by can could did do does doing down during each few for from further had has have having
    he her here hers herself him himself his how i if in into is it its itself just let me more most my myself
    no nor not now of off on once only or other our ours ourselves out over own same she should so some such
    than that the their theirs them themselves then there these they this those through to too under until up
    very was we were what when where which while who whom why will with would you your yours yourself
    yourselves get got make made want need like know think really thing things one two use using used way
    please thanks thank ok okay sure yes yeah hi hello hey help can't don't i'm it's that's what's
    tell give show explain question something anything would could should also still even well much many
    good great best better new way lot bit quick quickly maybe actually basically just right going go
    unrelated btw anyway switching gears topic another different separate side note also
    yaar kya hai hain ho kaise karu karna karte kar mujhe mera meri mere ka ki ke ko se mein bhi nahi
    aur toh na bata batao samjhao chahiye accha acha haan theek thik ek wala wali
    here's sure, let's step steps first next then example try check following note
    """.split()
)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 1 and not t.isdigit()]


class _Stripper(HTMLParser):
    BLOCK = {"p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4", "pre"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        self.parts.append(data)


def strip_html(html: str) -> str:
    s = _Stripper()
    s.feed(html)
    text = unescape("".join(s.parts)).replace("\xa0", " ")
    return re.sub(r"\n\s*\n+", "\n", text).strip()

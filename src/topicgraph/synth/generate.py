"""Single-topic segment generators.

`TemplateGenerator` is offline and deterministic: good for tests and for
exercising the plumbing, but its chats are far easier than real ones. Report
eval numbers from `LLMGenerator`, ideally backed by a different model family
than the one that runs the pipeline.
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
from pathlib import Path
from typing import Protocol

from ..llm import LLM, complete_json
from .topics import Topic

PERSONAS = [
    "an undergraduate student",
    "a product manager at a startup",
    "a backend software engineer",
    "a PhD researcher",
    "a working parent",
    "a freelance designer",
]
STYLES = [
    "short and direct, mostly lowercase, skips punctuation sometimes",
    "gives a few lines of background about their situation before asking",
    "casual, the occasional typo, writes the way people text",
    "a bit frustrated because a first attempt did not work",
    "polite, full sentences, but not formal",
]
# The kinds of things people actually ask assistants. One is picked per segment
# so the set covers all of them.
QUESTION_TYPES = [
    "how-to: wants step-by-step instructions to do something",
    "troubleshooting: something is broken or not working and they describe what they see",
    "explanation: wants to understand why or how something works",
    "comparison: deciding between two or three specific options",
    "recommendation: asks what they should pick or do in their situation",
    "planning: wants help putting together a plan or schedule",
    "quick fact: a short factual question, then a follow-up or two",
    "feedback: pastes or describes their own draft, numbers or idea and asks for review",
    "brainstorming: wants ideas or options to choose from",
    "sanity check: already has an answer in mind and wants it confirmed",
]

Exchange = tuple[str, str]  # (user message, assistant reply)


class SegmentGenerator(Protocol):
    name: str

    def generate(self, topic: Topic, n_turns: int, persona: str, style: str, rng: random.Random) -> list[Exchange]: ...


_USER_STEMS = [
    "how do I deal with {a} and {b}?",
    "can you explain {a} in the context of {b}",
    "my {a} keeps causing problems with {b}, what should I do",
    "what's the right way to handle {a} when {b} is involved?",
    "is it normal that {a} affects {b}?",
    "{a} vs {b} — which matters more here?",
    "any tips on {a}? also confused about {b}",
]
_FOLLOWUPS = [
    "thanks! can you make that shorter?",
    "ok and then?",
    "got it, thank you",
    "can you give an example?",
    "hmm, why?",
]
_ANSWER = [
    "Good question. With {a}, the key thing is how it interacts with {b}.",
    "Start by checking {a}; most issues with {b} come from there.",
    "In short, {a} and {b} are related: changing one usually affects the other, so look at {c} too.",
    "A common approach is to handle {a} first, then revisit {b} and {c}.",
    "Here is what I would do: review {a}, compare it with {b}, and keep an eye on {c}.",
]
_GENERIC = "overall in general usually typically important common approach depends situation option".split()


class TemplateGenerator:
    name = "template"

    def generate(self, topic: Topic, n_turns: int, persona: str, style: str, rng: random.Random) -> list[Exchange]:
        kw = list(topic.keywords)
        out: list[Exchange] = []
        for i in range(n_turns):
            if i > 0 and rng.random() < 0.15:
                user = rng.choice(_FOLLOWUPS)
            else:
                a, b = rng.sample(kw, 2)
                user = rng.choice(_USER_STEMS).format(a=a, b=b)
                if style.startswith("short"):
                    user = user.lower().rstrip("?")
            a, b, c = rng.sample(kw, 3)
            sentences = [rng.choice(_ANSWER).format(a=a, b=b, c=c) for _ in range(rng.randint(2, 3))]
            sentences.append(f"This is {rng.choice(_GENERIC)} {rng.choice(_GENERIC)} for {rng.choice(kw)}.")
            out.append((user, " ".join(sentences)))
        return out


class LLMGenerator:
    """Writes realistic exchanges with an LLM. Results are cached on disk so reruns are free."""

    SYSTEM = (
        "You write realistic transcripts of a real person chatting with an AI assistant. They are test "
        "data for software that detects topic changes in chat histories, so they must read like logs "
        "from a real product, not like polished examples.\n\n"
        "The user:\n"
        "- Writes like an ordinary person typing quickly: not polished, not textbook phrasing, no "
        "greetings, does not thank the assistant every turn.\n"
        "- Is not vague either: the first message has enough concrete detail (a version, an amount, "
        "a city, a date, an error line, what they already tried) that a reader knows what it is about.\n"
        "- Varies message length: some are a few words, some are several sentences.\n"
        "- In follow-ups, reacts to the previous answer the way people do: 'that didn't work', "
        "'what about the second one', 'ok and if I'm on a mac?', and does not restate the whole topic.\n"
        "- Stays on the given topic for the whole exchange. No tangents, no switching subjects.\n\n"
        "The assistant replies like a real assistant: helpful and specific, usually 50 to 200 words, "
        "using short lists, steps or code only when they fit. English only."
    )
    SCHEMA = {
        "type": "object",
        "properties": {
            "turns": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"user": {"type": "string"}, "assistant": {"type": "string"}},
                    "required": ["user", "assistant"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["turns"],
        "additionalProperties": False,
    }

    def __init__(self, llm: LLM, cache_path: str | Path | None = None):
        self.llm = llm
        self.name = f"llm:{llm.name}"
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[str, list[Exchange]] = {}
        self._lock = threading.Lock()
        if self.cache_path and self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self.cache[row["key"]] = [tuple(t) for t in row["turns"]]

    def generate(self, topic: Topic, n_turns: int, persona: str, style: str, rng: random.Random) -> list[Exchange]:
        variant = rng.randint(0, 10**9)
        key = hashlib.sha1(f"{self.llm.name}|{topic.id}|{n_turns}|{persona}|{style}|{variant}".encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        qtype = rng.choice(QUESTION_TYPES)
        prompt = (
            f"Topic: {topic.name} ({topic.brief}).\n"
            f"User: {persona}. Writing style: {style}.\n"
            f"Kind of question: {qtype}.\n"
            f"Write exactly {n_turns} user/assistant exchange(s)."
        )
        turns = complete_json(self.llm, self.SYSTEM, prompt, self.SCHEMA)["turns"][:n_turns]
        result = [(t["user"].strip(), t["assistant"].strip()) for t in turns]
        if len(result) < n_turns:
            raise RuntimeError(f"generator returned {len(result)} turns, wanted {n_turns}")
        with self._lock:
            self.cache[key] = result
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with self.cache_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": key, "turns": result}, ensure_ascii=False) + "\n")
        return result


class TransitionRewriter(Protocol):
    def rewrite(self, previous: Exchange, boundary_user: str, rng: random.Random) -> str: ...


_CUES = ["Unrelated, but ", "btw, ", "Switching gears — ", "Different question: ", "Also, ", "Random q: ",
         "oh also ", "Another thing — "]


class TemplateTransitions:
    """Prefixes a cue phrase at most boundaries; some switches stay abrupt, as in real chats."""

    def __init__(self, p_cue: float = 0.6):
        self.p_cue = p_cue

    def rewrite(self, previous: Exchange, boundary_user: str, rng: random.Random) -> str:
        if rng.random() >= self.p_cue:
            return boundary_user
        msg = boundary_user[:1].lower() + boundary_user[1:]
        return rng.choice(_CUES) + msg


class LLMTransitions:
    SYSTEM = (
        "You edit one user message in an AI-assistant chat so the topic switch reads naturally, the way "
        "people really switch topics mid-chat: sometimes a signal like 'unrelated, but…', 'oh also', "
        "'different question', and often no signal at all, just the new question. Change as little as "
        "needed. The message must stay entirely about its own subject: do not mention or continue the "
        "previous topic. Keep the user's style. English only. Reply with the edited message only."
    )
    SCHEMA = {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
        "additionalProperties": False,
    }

    def __init__(self, llm: LLM):
        self.llm = llm

    def rewrite(self, previous: Exchange, boundary_user: str, rng: random.Random) -> str:
        prompt = (
            f"<previous_exchange>\nUser: {previous[0]}\nAssistant: {previous[1][:600]}\n</previous_exchange>\n"
            f"<message_to_edit>\n{boundary_user}\n</message_to_edit>"
        )
        return complete_json(self.llm, self.SYSTEM, prompt, self.SCHEMA)["message"].strip()

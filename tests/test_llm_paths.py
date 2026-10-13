"""LLM-backed steps, exercised with a fake LLM so no network or key is needed."""

import json
import random

from topicgraph.models import Chat, Message
from topicgraph.pipeline import build_graph
from topicgraph.synth.generate import LLMGenerator, LLMTransitions
from topicgraph.synth.topics import BY_ID


class FakeLLM:
    name = "fake"

    def __init__(self):
        self.calls = []

    def complete(self, system, prompt, schema=None):
        self.calls.append(system)
        props = set(schema["properties"])
        if props == {"new_topic"}:
            return json.dumps({"new_topic": "visa" in prompt.split("<new_turn>")[1]})
        if props == {"topic"}:
            return json.dumps({"topic": "Visa documents" if "visa" in prompt else "Python errors"})
        if props == {"turns"}:
            n = int(prompt.split("exactly ")[1].split()[0])
            return json.dumps({"turns": [{"user": f"q{i}", "assistant": f"a{i}"} for i in range(n)]})
        if props == {"message"}:
            return json.dumps({"message": "oh also, " + prompt.split("<message_to_edit>\n")[1].split("\n")[0]})
        raise AssertionError(props)


def test_pipeline_with_llm_judge_and_labels():
    msgs = [
        Message("user", "python keyerror"), Message("assistant", "missing dict key"),
        Message("user", "ok"), Message("assistant", "anything else?"),
        Message("user", "what about the visa"), Message("assistant", "bring documents"),
    ]
    llm = FakeLLM()
    graph = build_graph([Chat("c", "Python", "claude", msgs)], llm=llm)
    assert [(s.start, s.end) for s in graph.segments] == [(0, 2), (2, 3)]
    assert {s.label for s in graph.segments} == {"Python errors", "Visa documents"}


def test_llm_generator_caches(tmp_path):
    llm = FakeLLM()
    cache = tmp_path / "cache.jsonl"
    gen = LLMGenerator(llm, cache)
    out = gen.generate(BY_ID["python_errors"], 3, "p", "s", random.Random(1))
    assert len(out) == 3
    again = LLMGenerator(llm, cache).generate(BY_ID["python_errors"], 3, "p", "s", random.Random(1))
    assert again == out and len(llm.calls) == 1


def test_llm_transitions():
    msg = LLMTransitions(FakeLLM()).rewrite(("u", "a"), "visa question", random.Random(0))
    assert msg == "oh also, visa question"


def test_parallel_generation_is_deterministic():
    from topicgraph.synth.stitch import build_eval_set

    def run(workers):
        llm = FakeLLM()
        chats = build_eval_set(20, LLMGenerator(llm), seed=3, transitions=LLMTransitions(llm), workers=workers)
        return [(c.case, c.turns, [(s.start, s.end, s.topic) for s in c.segments]) for c in chats]

    assert run(1) == run(6)

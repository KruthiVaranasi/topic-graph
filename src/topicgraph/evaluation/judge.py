"""LLM judge for the real-data sanity run, where no ground truth exists."""

from __future__ import annotations

import random
from typing import Any

from ..llm import LLM, complete_json
from ..models import Segment

SYSTEM = (
    "You audit the output of a tool that splits AI-assistant chats into topic segments and names them. "
    "Judge the excerpt strictly and independently of how the tool might have reasoned."
)
SCHEMA = {
    "type": "object",
    "properties": {
        "single_topic": {"type": "boolean", "description": "The excerpt stays on one topic throughout."},
        "label_fits": {"type": "boolean", "description": "The topic name accurately describes the excerpt."},
        "node_fits": {"type": "boolean", "description": "The excerpt belongs under the broader topic node."},
    },
    "required": ["single_topic", "label_fits", "node_fits"],
    "additionalProperties": False,
}


def judge_sample(segments: list[Segment], node_labels: dict[int, str], llm: LLM, n: int = 50,
                 seed: int = 0, max_chars: int = 5000) -> dict[str, Any]:
    rng = random.Random(seed)
    sample = rng.sample(segments, min(n, len(segments)))
    rows = []
    for s in sample:
        prompt = (
            f"<segment_label>{s.label}</segment_label>\n<topic_node>{node_labels.get(s.node_id, '')}</topic_node>\n"
            f"<excerpt>\n{s.text[:max_chars]}\n</excerpt>"
        )
        rows.append(dict(complete_json(llm, SYSTEM, prompt, SCHEMA), id=s.id, hidden=s.hidden))
    rate = lambda k: round(sum(r[k] for r in rows) / len(rows), 4) if rows else None  # noqa: E731
    return {"n": len(rows), "single_topic": rate("single_topic"), "label_fits": rate("label_fits"),
            "node_fits": rate("node_fits"), "rows": rows}

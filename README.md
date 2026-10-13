# Topic Graph for AI chats

AI chat titles come from the first prompt and never change, so anything discussed later in a chat is hard to find. Topic Graph is a local tool that splits your ChatGPT, Claude and Gemini history into **topic segments** and links them into **one cross-tool graph**. Each topic node lists every segment about it, from every tool, including the ones buried under unrelated titles.

The full product spec (problem, users, metrics, eval design, risks) is in [`docs/PRD.md`](docs/PRD.md).

## Quick start

```bash
pip install -e ".[dev]"            # numpy is the only runtime dependency

# 1. Build the graph from your exports (files or folders, any mix of tools)
topicgraph build ~/Downloads/chatgpt-export ~/Downloads/claude-export ~/Downloads/Takeout --out out
open out/index.html                # self-contained, works offline

# 2. "Have I solved this before?"
topicgraph search "schengen visa documents" --graph out/graph.json
```

Where the exports come from:

| Tool | How to export | File the importer reads |
|---|---|---|
| ChatGPT | Settings → Data controls → Export data | `conversations.json` or `chat.html` |
| Claude | Settings → Privacy → Export data | `conversations.json` |
| Gemini | Google Takeout → "My Activity" → Gemini Apps (choose JSON) | `My Activity/Gemini Apps/MyActivity.json` (HTML is parsed best-effort) |

Gemini's Takeout has no conversation ids, so chats are rebuilt as sessions: prompts less than 30 minutes apart form one chat.

## Privacy

By default everything runs on your machine: the embedder is pure numpy, and the viewer is a single HTML file with no CDN or network calls. Chat content never leaves the computer.

`--llm claude` is opt-in and **sends segment text to the Anthropic API** for boundary checks and topic names. It gives better results but breaks the "nothing leaves the machine" guarantee, so use it knowingly (or only on the synthetic eval set). `--embedder st` downloads a sentence-transformers model once, then runs locally.

## How it works

```
exports ──► import ──► segment ──► label ──► cluster ──► graph + viewer
            (per tool)  (per chat)  (per segment) (across all chats and tools)
```

1. **Import** (`importers/`): one thin module per tool turns its export into `Chat` objects. ChatGPT exports are trees (edits and regenerations branch), so the importer follows the branch you last saw.
2. **Segment** (`segment.py`): each turn (user message + reply) is scored by its similarity to the previous turn and to the running segment centroid. Low scores start a new segment. Scores in the borderline band go to an LLM judge when `--llm` is set, otherwise they are cut at a calibrated midpoint. Short follow-ups ("thanks, make it shorter") stay in the current segment; cue phrases ("unrelated, but…", "btw") push towards a cut.
3. **Label** (`label.py`): top TF-IDF keywords offline, or a 2–5 word name from the LLM.
4. **Cluster** (`cluster.py`): average-linkage agglomerative clustering over segment vectors (plus label vectors when LLM labels exist), stopping at a similarity threshold, so the number of topics is not fixed in advance.
5. **Graph** (`pipeline.py`, `viewer.py`): nodes are topics; `cooccur` edges join topics that share a chat and `follows` edges join a topic to the one that came right after it. Clicking a node lists its segments with tool, date, chat title, turn range and a link back to the chat. Segments that are not the chat's opening topic are tagged **hidden**: a title-based organizer cannot find them.

`build` prints the **hidden-topic rate**: the share of topic nodes that never open a chat, so they live only under other titles.

## Evaluation

The eval set is synthetic: drifting chats are built by stitching single-topic segments together, so the true boundaries and topics are known by construction (design in [`docs/PRD.md`](docs/PRD.md#evaluation-with-synthetic-data)).

```bash
topicgraph synth --out data/synth --n 300                    # offline template chats
topicgraph synth --out data/synth-llm --n 300 --llm claude   # LLM-written chats (cached in data/synth-llm/cache)
topicgraph eval  --data data/synth --out metrics.json        # score against truth.json
topicgraph eval  --data data/synth --sweep                   # calibrate thresholds
topicgraph stats data/synth/exports ~/Downloads/claude-export   # distribution check vs real exports
topicgraph build ~/exports --out out --llm claude --judge-sample 50   # real-data sanity run with an LLM judge
```

What `synth` builds:

- 40 topics across coding, career, travel, health, finance, cooking, study and life, with 10 deliberate near-neighbour pairs (Python errors vs. Python packaging, Schengen vs. US visa, …) in `synth/topics.py`.
- The case mix from the spec: 40% clean switches, 20% near-neighbours, 15% returns to an earlier topic, 15% one-turn asides, 10% single-topic chats.
- Personas, writing styles and 10 question types (how-to, troubleshooting, explanation, comparison, recommendation, planning, quick fact, feedback, brainstorming, sanity check), written to read like real users: not polished, not vague. English only. Each chat is titled from its first segment, like ChatGPT.
- A transition rewrite at every boundary that changes only the user side of the first turn after the cut, so ground truth stays valid.
- Real export formats for all three tools plus `truth.json`, so every eval also tests the importers.

Metrics (`evaluation/metrics.py`):

| Metric | Definition | Target |
|---|---|---|
| Topic recall | A true segment counts as found when ≥ 50% of its turns sit under its topic's node. The topic's node is the single node holding most of its turns, so a topic split across two nodes loses recall. | ≥ 0.90 |
| Topic precision | A predicted segment is correct when its majority true topic matches its node's majority topic. | ≥ 0.80 |
| Boundary F1 | Predicted vs. true topic shifts, ±1 turn tolerance. | ≥ 0.75 |
| Hidden recall | Topic recall on segments that are not the chat's first segment. | reported |
| One-turn recall | Topic recall on one-turn asides, the hardest hidden topics. | reported |

The report also breaks results down per case, and its import check counts missing chats and turn-count mismatches.

### Current results

**Realistic eval set: `evalsets/llm-v1`.** 300 chats and 1,100 segments, written by LLMs following [`WRITING_GUIDE.md`](evalsets/llm-v1/WRITING_GUIDE.md). Each segment was routed by difficulty: 223 simple ones to Claude Haiku, 322 medium ones to Claude Sonnet, and 555 complex or most human-like ones to Claude Opus. Opus then rewrote the 800 topic-switch messages ([`TRANSITION_GUIDE.md`](evalsets/llm-v1/TRANSITION_GUIDE.md)). Rebuild the export files with `topicgraph batch assemble --out evalsets/llm-v1`.

Thresholds were tuned on the even-numbered chats (`--split tune`) and scored on the odd-numbered ones (`--split test`). Offline mode only: no LLM in the pipeline.

| Metric (test half, 150 chats) | Hashing embedder | LSA embedder | Target |
|---|---|---|---|
| Topic recall | 0.603 | 0.494 | ≥ 0.90 |
| Topic precision | 0.655 | **0.839** ✓ | ≥ 0.80 |
| Boundary F1 | 0.561 | 0.621 | ≥ 0.75 |
| Hidden recall | 0.597 | 0.472 | |
| One-turn recall | 0.400 | 0.280 | |

LSA settings: `--embedder lsa --low 0.08 --mid 0.17 --high 0.44 --merge 0.3 --cluster-threshold 0.35`. This is the configuration that follows the product rule (best recall with precision ≥ 0.80). The hashing column shows the best recall/precision balance, because no hashing setting reaches the precision floor.

**What this says.** Word-based methods plateau well short of the recall target on realistic chats. People don't repeat keywords, so one topic gets split across many nodes: 241 predicted nodes for 40 true topics. That is a recall failure, because a user opening one node misses the others. Loosening the clustering merges unrelated topics and breaks the precision floor. Closing the gap needs semantic similarity: a sentence-embedding model (`--embedder st`, local after a one-time download, which this build environment blocks) and/or the LLM boundary judge and labels (`--llm`). Those are the next experiments.

For comparison, the offline template set (`topicgraph synth`, no LLM) scores recall 0.894, precision 0.887 and boundary F1 0.733. Template chats reuse a small vocabulary per topic, so treat them as a plumbing test only.

## Using a different model family

Every LLM step takes an object with `complete(system, prompt, schema) -> str` (`llm.py`). `ClaudeLLM` is bundled; select a model with `--llm claude:<model-id>` (default `claude-opus-5-5`, effort `low`). To generate the eval data with one family and run the pipeline with another, as the spec requires, add a ~20-line adapter for the other vendor's SDK with the same `complete` method and pass it to `LLMGenerator`.

## Project layout

```
src/topicgraph/
  importers/      chatgpt.py, claude.py, gemini.py — one thin module per export format
  segment.py      topic-shift detection (+ optional LLM boundary judge)
  label.py        keyword or LLM topic names
  cluster.py      agglomerative clustering (numpy)
  pipeline.py     build_graph, edges, search, save
  viewer.py       offline HTML graph viewer
  synth/          topics, segment generators, stitching, export writers
  evaluation/     metrics, distribution stats, LLM judge
  cli.py          build | search | synth | eval | stats
tests/            importers, pipeline, metrics, LLM paths (with a fake LLM)
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

Out of scope for the MVP: live sync with the assistants, renaming chats inside the tools, sharing graphs.

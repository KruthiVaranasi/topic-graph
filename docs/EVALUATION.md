# Does Topic Graph help someone find an old chat?

The README's segment metrics check whether the pipeline splits and groups chats correctly. That is not the user's job. The user's job is "find the chat where I talked about X", so this page tests that job directly, against what people already have, and records what it costs to run.

Everything here runs on the 150 held-out test chats of `evalsets/llm-v1` (AI-written, no personal history). Commands are at the end.

## 1. Can a user find a chat whose title says something else?

**Why it matters.** This is the problem the product exists for. If today's tools already find these chats, Topic Graph has no reason to exist.

**How it was tested.** For each of the 40 topics there are two queries in [`evalsets/llm-v1/queries.json`](../evalsets/llm-v1/queries.json): a short keyword query ("schengen visa documents") and a *remembered* one that avoids the topic's own words ("paperwork for my europe trip appointment"), because people rarely remember the exact words they typed months ago. The queries were written from the topic list alone, without reading any chat. A chat counts as found if it appears in the first 20 results; for browsing, if it is listed under the topic the user opens. On average each topic appears in 13 test chats, 9 of them under a title about something else (*hidden*).

**Result: hidden chats found, remembered queries.**

| How the user looks | Hidden chats found | What they skim |
|---|---|---|
| Chat titles (the sidebar today) | 0% | — |
| Keyword search, all words must match (how the apps' own search behaves) | 16% | 20 results |
| Keyword search over full chat text, ranked (BM25), all apps at once | 58% | 20 results |
| **Topic Graph search** (Sentence-BERT) | **74%** | 20 results |
| Meaning search over individual messages, no topic graph | 78% | 20 results |
| **Topic Graph: open the top 3 matching topics** | **83%** | about 26 chats, 55% of them relevant |

With keyword queries every method does better (Topic Graph search 89%, browsing the top 3 topics 87%, ranked keyword search 82%), but the order is the same.

**What it means.**
- The problem is real: titles find none of the hidden chats, and exact-word search finds 16% once the user's wording drifts.
- Topic Graph finds 74–83%. The gain comes from **matching meaning instead of words**: plain meaning search over messages does as well for search. The topic graph earns its place when the user **browses** a topic without phrasing a query, and for the jobs search can't do (seeing every topic across tools at once).
- The ranked keyword baseline is generous: it searches all three apps in one index, which no app offers today.

**Decision.** Lead the product with meaning search plus topic browsing, and make Sentence-BERT the default (section 3).

## 2. What does it cost to run?

**Why it matters.** The tool runs on the user's laptop. A first build that takes an hour, or a search that takes seconds, breaks "have I asked this before?".

**How it was tested.** One laptop CPU (no GPU), all 300 test chats (5,156 turns, about 4 million characters).

| Step | Measured | For a user |
|---|---|---|
| Build, hashing embedder | 16 s for 300 chats | instant |
| Build, Sentence-BERT | 8.3 min for 300 chats (1.7 s per chat) | a heavy user with 2,000 chats would wait about an hour, once (projected) |
| Clustering, 1,000 / 2,000 / 4,000 segments | 0.5 s / 4.3 s / 34 s (about 8× per doubling) | 2,000 chats is about 8,000 segments: roughly 4–5 min and 0.5 GB just for clustering (projected) |
| Search, before this change | 3.2 s per query (every segment re-embedded) | too slow for a quick check |
| Search with the index saved at build time | 11 ms per query (vs 0.5 s re-embedding, on a 234-segment graph) | instant |

**Decision.** Save the search index at build time (done). Next: incremental builds so a new export only processes new chats, and a faster clustering method above ~2,000 chats.

## 3. Do users get the setup the results are based on?

**Why it matters.** The README reported Sentence-BERT results, but `topicgraph build` with no flags used the hashing embedder with thresholds tuned for template chats.

**Result (150 test chats).**

| Setup | Recall | Precision | Groups for 40 topics |
|---|---|---|---|
| Old default (hashing) | 0.58 | 0.67 | 286 |
| Sentence-BERT with the hashing thresholds | 0.91 | **0.14** | 5 |
| Sentence-BERT with its own thresholds | 0.75 | 0.82 | 128 |

The middle row is why recall needs a precision floor: lumping everything into 5 groups "finds" 91% of topics and is useless.

**Decision.** `--embedder auto` is now the default: Sentence-BERT when installed, with the thresholds tuned for it; the numpy-only hashing embedder otherwise, with a message saying how to get better results.

## What this does not show yet

- **Real histories.** The chats are AI-written and may be cleaner than real ones. The next step is a run on real exports, reporting only totals.
- **Other people.** The problem evidence is one person's experience plus this test.
- **The optional LLM mode** (`--llm claude`) was not evaluated here.

## Reproduce

```bash
topicgraph batch assemble --out evalsets/llm-v1      # rebuild the export files (once)
topicgraph eval-search --data evalsets/llm-v1        # section 1
topicgraph eval --data evalsets/llm-v1 --split test  # section 3, default settings
```

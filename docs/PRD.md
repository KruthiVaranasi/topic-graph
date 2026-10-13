# Topic Graph for AI chats — product spec

## Problem

AI chat titles are set from the first prompt and never change, so topics discussed later in a chat become unfindable.

- **Topic drift is normal.** A chat titled "Python error" can later cover a visa question and a recipe. In a ChatGPT HTML export, all three sit under the original title.
- **Existing organizers inherit the error.** Folder and auto-categorize tools (e.g. MindMarks, Cleanest AI) label whole chats, so the drifted topics stay hidden.
- **Knowledge is split across tools.** People use ChatGPT, Claude, Gemini and Copilot. Each organizes only its own history, chronologically.

Result: users re-ask questions they already solved and cannot see what they have actually been thinking about.

## Users and jobs to be done

Primary user: heavy AI-assistant users (students, PMs, engineers, researchers) with hundreds of chats across two or more tools.

| Job | Today | With topic graph |
|---|---|---|
| "Find everything I discussed about X" | Keyword search per tool; misses chats titled otherwise | One topic node lists every segment, from every tool |
| "What have I been thinking about lately?" | Scroll a chronological sidebar | See topic clusters and how they grew over time |
| "Don't make me solve this twice" | Re-ask the assistant | Surface the earlier segment before re-asking |

## Solution and MVP scope

A local tool that splits chats into topic segments and links them into one cross-tool graph.

Pipeline:

1. **Import** chat exports: ChatGPT (HTML/JSON), Claude data export, Gemini via Google Takeout.
2. **Segment** each chat where the topic shifts (embedding similarity between consecutive turns, then an LLM check on borderline breaks).
3. **Label** each segment with a short topic name.
4. **Cluster** labels across all chats and tools into topic nodes.
5. **Graph** it: nodes are topics; edges join topics that often co-occur or follow each other. Clicking a node lists its segments with source tool, date and link.

In scope for MVP: the three exports above, single user, runs locally, read-only.
Out of scope: live sync with the assistants, renaming chats inside the tools, sharing graphs with others.

## Success metrics

North-star metric: **topic recall, with a precision floor of 80%.**

Why recall comes first. For "find everything about X," a missed segment is invisible: the user assumes it never existed. An extra, irrelevant segment is visible and costs a two-second skim. The product's core claim is recovering hidden topics, and recall measures exactly that. Precision still has a floor, because a topic node that is half noise loses trust.

| Metric | What it measures | Target |
|---|---|---|
| Topic recall | Share of true segments on a topic that appear under its node | ≥ 90% |
| Topic precision | Share of segments under a node that truly belong | ≥ 80% |
| Boundary F1 | How accurately topic-shift points are found (±1 turn tolerance) | ≥ 0.75 |
| Hidden-topic rate | Share of topics that exist only inside chats titled otherwise | Reported, not targeted |

The hidden-topic rate is the headline number for the case study: it sizes the problem the product solves.

One exception: if the tool later suggests renaming chats, that feature favors precision, because a wrong title is visible and irritating.

## Evaluation with synthetic data

The eval set is generated, not hand-labelled: we build drifting chats ourselves, so the true topic boundaries and labels are known by construction.

How the set is built:

1. **Define a topic list.** About 40 topics typical of assistant use (coding, career, travel, health, finance, cooking, study), with a few near-neighbours on purpose (Python errors vs. Python packaging).
2. **Generate single-topic segments.** An LLM writes short user–assistant exchanges for each topic, 2–8 turns, in varied personas and styles, including Hinglish.
3. **Stitch segments into chats.** Combine 2–5 segments per chat. Every boundary position and topic label is saved as ground truth. The chat title is taken from the first segment, mimicking ChatGPT.
4. **Add realistic transitions.** An LLM rewrites the turn at each boundary so the switch reads naturally ("unrelated, but…"), without changing which side of the boundary each turn is on.
5. **Spread chats across tools.** Format outputs as ChatGPT, Claude and Gemini exports, so the importer is tested too.

Target mix of about 300 chats:

| Case | Share | What it tests |
|---|---|---|
| Clean switches between unrelated topics | 40% | Baseline segmentation |
| Near-neighbour topics | 20% | Over-merging; precision |
| Return to an earlier topic in the same chat | 15% | Linking two segments to one node |
| One-turn asides | 15% | Recall on tiny segments, the hardest hidden topics |
| Single-topic chats (no drift) | 10% | False boundaries |

Keeping the synthetic data honest:

- Different model families generate the data and run the pipeline, so the segmenter is not grading its own style.
- A 20-chat spot check (about 30 minutes, not full labelling) confirms the stitched chats read like real ones.
- Distribution check against my real exports: turns per chat, message length, topics per chat.
- Real-data sanity run: apply the pipeline to my own history and report the hidden-topic rate, with an LLM judge rating a sample of segments.

## Risks and open questions

| Risk | Mitigation |
|---|---|
| Synthetic chats are cleaner than real ones, so scores look too good | Hard-case mix above; report real-data sanity results beside synthetic scores |
| Chats are deeply personal | Process locally; no upload of chat content |
| Platforms build native topic views | Cross-tool graph is the defensible part; no single assistant has reason to index its rivals |
| Export formats change | Thin importer per tool, covered by the synthetic export tests |

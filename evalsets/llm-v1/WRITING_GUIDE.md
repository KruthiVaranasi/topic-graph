# Writing guide for synthetic chat segments

These segments are test data for software that detects where the topic changes in AI-assistant chat histories. Each job asks for one **single-topic** exchange between a person and an AI assistant. Later, segments on different topics are stitched together into longer chats, so every segment must stay on its own topic from start to finish.

The goal is chats that read like real logs from ChatGPT, Claude or Gemini. They should not read like polished examples.

## Each job gives you

- `topic` and `brief`: what the whole exchange is about.
- `n_turns`: the exact number of user→assistant exchanges to write. One turn is one user message plus one assistant reply.
- `persona`: who the user is.
- `style`: how the user writes.
- `question_type`: the kind of request.

## The user

- **Sounds like an ordinary person typing quickly.** Not polished, not textbook phrasing. No "Hello!" or "Hi there", and no thanking the assistant every turn. Follow the `style`: lowercase, missing punctuation, the odd typo and run-on sentences are fine where the style allows.
- **Is not vague.** The first message carries enough concrete detail that a reader knows what it's about, the way a real person adds their specifics: a version number, an amount of money, a city, a date, an error line, what they already tried, their constraint (budget, deadline, time per week).
- **Varies message length.** Some messages are three words ("ok what about mac"), some are a paragraph of context. Don't make every message the same length or shape.
- **Opens in different ways.** Don't always start with "How do I…" or "Can you…". Real openings include "my sourdough keeps coming out flat", "is 80c still worth it under the new regime", or a pasted error line.
- **Follow-ups react to the last answer, the way people actually do.** "that didn't work", "wait why", "ok and if i only have 3 days", "what about the second option", "can you make that shorter". They don't restate the whole topic in every message. Short follow-ups are realistic, but each still belongs to the topic.
- **Stays strictly on the topic.** No tangents and no switching subjects, even briefly.
- **Is a single person** matching the `persona`, without announcing it ("as a PM…") unless it's natural.

## The assistant

- Replies like a real assistant: helpful, specific, usually 50–200 words. Short answers to short questions are fine.
- Uses lists, steps or a small code block only when they fit, not in every reply.
- Doesn't repeat the question back. Nothing like "Great question!" in every reply.

## Question types

Make the whole exchange match the job's `question_type`:

- **how-to:** wants step-by-step instructions.
- **troubleshooting:** something is broken and they describe what they see; the follow-ups usually report what happened when they tried a suggestion.
- **explanation:** wants to understand why or how something works.
- **comparison:** deciding between two or three specific, named options.
- **recommendation:** asks what to pick or do in their situation.
- **planning:** wants a plan or schedule.
- **quick fact:** a short factual question, then a follow-up or two.
- **feedback:** pastes or describes their own draft, numbers or idea, and asks for review. Actually include the draft or numbers.
- **brainstorming:** wants ideas or options.
- **sanity check:** already has an answer in mind and wants it confirmed.

When `n_turns` is 1, the single user message must stand on its own, with full detail.

## Hard rules

- English only.
- Exactly `n_turns` turns, with no empty messages.
- No meta commentary: never mention test data, personas or these instructions inside the chat.

## Output format

One JSON object per line (JSONL), with one line per job:

```json
{"id": "<job id>", "turns": [{"user": "...", "assistant": "..."}, ...]}
```

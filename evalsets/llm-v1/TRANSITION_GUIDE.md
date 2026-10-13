# Writing guide for topic-switch messages

Segments on different topics are stitched together into one chat. Each job is the first user message of a new topic, plus the exchange that came right before it (on a different topic). Edit the message so the switch reads the way real people change topic mid-chat.

## How people actually switch

- **Often with no signal at all.** They just ask the new thing. Leave a good share of messages (roughly 40%) unchanged or nearly unchanged.
- **Sometimes with a light signal.** "unrelated but", "oh also", "different question", "btw", "ok switching gears", "random q", "another thing".
- **Sometimes by closing the previous topic briefly first.** "ok that worked, thanks. separately…" or "cool. now something totally different:".
- **Varied.** Don't use the same signal twice in a row.

## Rules

- The edited message must stay entirely about its **own** topic. Never mention, continue or connect to the previous topic beyond a short closing like "thanks, that helped".
- Keep every concrete detail of the original message (numbers, names, the actual question), and keep the user's writing style and casing.
- Change as little as needed. English only.

## Output format

One JSON object per line (JSONL), one line per job:

```json
{"id": "<job id>", "message": "<edited message>"}
```

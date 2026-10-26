# harness

Python package for the harness engineering workshop.

## Setup

```bash
cp .env.example .env   # fill in HARNESS_BEDROCK_API_KEY
uv sync
```

## Run

```bash
uv run harness "Build a landing page for a coffee shop"   # one prompt, no tools
uv run python demos/day1_dice.py                          # day 1 tool-calling demo
```

Switch models with `HARNESS_BEDROCK_MODEL` (default `deepseek.v3.2`), e.g.
`HARNESS_BEDROCK_MODEL=us.anthropic.claude-sonnet-4-5-20250929-v1:0`.


## Curriculum

| Day | start Tag | End tag |
| --- | --- | --- |
| day-1 | `git checkout v1.start` | `git checkout v1` |
| day-2 | `git checkout v2.start` | `git checkout v2` |


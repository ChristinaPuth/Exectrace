# Company Leadership RAG Agent

Given a company domain, collects its leadership (C-level, VPs, Heads) from
public sources into a verifiable SQLite dataset, and answers questions over it
through an LLM chat.

Status: **milestone M0** (project setup). See [SPEC.md](SPEC.md) for the design
and `SPEC_checklist.xlsx` for the requirement checklist.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Copy `.env.example` to `.env` and fill in at least one of `OPENAI_API_KEY` or
`GROQ_API_KEY`. Then verify the keys:

```bash
uv run leadership check
```

This makes real calls: it lists the models each key can use and tests JSON
output and tool calling on the models named in `config.toml`.

## Tests

```bash
uv run pytest
```

The default run needs no network or keys.

## Configuration

- `.env` — secrets only. Never committed.
- `config.toml` — models, collection limits, agent limits, eval thresholds.

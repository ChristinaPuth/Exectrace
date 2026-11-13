# Company Leadership RAG Agent

Given a company domain, collects its leadership (C-level, VPs, Heads) from
public sources into a verifiable SQLite dataset, and answers questions over it
through an LLM chat.

Status: **M0** (project setup) and **M1** (database schema, role resolution
rules) and **M2** (finding and fetching sources) are done and accepted.
Extracting people from the sources, and the chat, are not built yet. See [SPEC.md](SPEC.md) for
the design, `SPEC_checklist.xlsx` for the requirement checklist, and
`acceptance/` for each milestone's test results.

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

## Collect sources

```bash
uv run leadership collect https://robinhood.com/
```

Finds and fetches the company's public leadership sources: its own pages,
SEC filings if it is a public company, and web search results. Pages are
cached, so a second run makes no network calls; add `--refresh` to fetch
again. Each run prints its usage against the limits in `config.toml` and
writes a log to `logs/`. To see what was stored:

```bash
uv run python tools/show_run.py robinhood.com
```

Optional keys in `.env`: `TAVILY_API_KEY` (web search), `SEC_USER_AGENT` (SEC
filings), `WIKIPEDIA_USER_AGENT` (Wikipedia). Each source is skipped, and the
run says so, when its key is missing.

Known limits of collection:

- **Team pages may be missing or unreadable.** Campfire has no team page.
  Robinhood's leadership page is on a site that refuses automated clients;
  it is recorded as `skipped` with the reason. The tool never imitates a
  browser to get past a refusal. Every run report ends with the list of
  pages that could not be read, the company's own pages first.
- **Wikipedia is not used.** Wikimedia requires a contact address, and none
  is configured.
- **Namesakes.** A search result on a site whose name resembles the
  company's is used only if the page names the company's own domain;
  otherwise it is recorded as `skipped` and never read for people.
- **Original sources only.** Machine-generated summary pages from search
  and AI companies (`tavily.com`, `exa.ai`, `vectorshift.ai`, `derrick-app.com`)
  are never used as
  evidence. The list is by site name, so a new such site is not caught.
- **Publication dates** come only from the page itself or the SEC's filing
  index. Many pages state none; their date is left empty.

## Tests

```bash
uv run pytest
```

The default run needs no network or keys.

## Configuration

- `.env` — secrets only. Never committed.
- `config.toml` — models, collection limits, agent limits, eval thresholds.

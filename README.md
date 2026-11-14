# Company Leadership RAG Agent

Give it a company's domain or URL. It finds the company's leadership
(C-level, vice presidents, heads of department) in public sources, stores
every fact with the source and the quote that support it, and answers
questions about it through an LLM chat that can only read that data.

The repository includes the collected data for both test companies,
**Campfire** (`meetcampfire.com`) and **Robinhood** (`robinhood.com`), so the
chat runs straight after cloning, with one LLM key and no collection step.

- [Quick start](#quick-start)
- [What is delivered](#what-is-delivered)
- [How it works](#how-it-works): data sourcing, storage design, agent architecture
- [Measured results](#measured-results)
- [Known limits](#known-limits)
- [How this was built with Claude Code](#how-this-was-built-with-claude-code)
- [Collecting a new company](#collecting-a-new-company)
- [Tests and configuration](#tests-and-configuration)

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11 or newer (uv
installs it if needed).

```bash
git clone https://github.com/ChristinaPuth/Exectrace.git
```

```bash
cd Exectrace
```

```bash
uv sync
```

Copy `.env.example` to `.env` and set **one** of `GROQ_API_KEY` or
`OPENAI_API_KEY`. That is all the chat needs.

**Web chat:**

```bash
uv run python -m streamlit run app.py
```

It opens at http://localhost:8501. Pick a company in the sidebar and ask, for
example "Who is the CEO?", "How many VPs do they have?", "Who heads
marketing?", "Where is their CEO based?" or "Who was the previous CFO?".
Under each answer, **SQL and source rows** shows the queries that ran and the
rows they returned.

**Command-line chat:**

```bash
uv run leadership chat robinhood.com
```

Type a question; `exit` stops. Ask one question with
`-q "Who is the CEO?"`, and add `--show-sql` to see the SQL.

(`python -m streamlit` rather than `streamlit`: some Windows security
policies block the `streamlit.exe` launcher. The tables on the page are
plain Markdown for the same reason: `st.dataframe` needs pyarrow's compiled
library.)

## What is delivered

| Deliverable | Where |
|---|---|
| Runnable repository | this repo: `uv sync`, one key, then the commands above |
| Collected data, both companies | `fixtures/leadership.db` (SQLite, what the chat reads), `fixtures/meetcampfire.com.json`, `fixtures/robinhood.com.json` (every person, role, claim, source and evidence quote) |
| Chat interface | web page (`app.py`, Streamlit) and command line (`leadership chat`) |
| Coding-agent session export | `session/` (Claude Code transcripts; see [below](#how-this-was-built-with-claude-code)) |
| Design | `SPEC.md` (design and decisions), `SPEC_checklist.xlsx` (173 requirements with status) |
| Evidence per milestone | `acceptance/M0.md` ... `M7.md`, each with its raw test log |
| Gold sets and evaluations | `fixtures/gold/` (hand-verified leadership), `fixtures/chat_eval.json` and `chat_eval_results.json` (chat evaluation), `fixtures/citation_review.md` (manual review of 10 answers) |

## How it works

```
domain ──► discover ──► fetch + cache ──► extract (LLM) ──► check evidence ──► claims
                                                                                │
                      chat (LLM + one read-only SQL tool) ◄── views ◄── rules ◄─┘
```

### 1. Data sourcing

- **Where it looks** (`leadership/collect.py`, `leadership/sources/`):
  - the company's own site: leadership, team and about pages, and its newsroom;
  - SEC EDGAR filings when the company is public (proxy statements, 8-Ks);
  - web search through Tavily: generic leadership queries, then one targeted
    search per uncertain role.

  LinkedIn is never fetched.
- **Sources are ranked by tier.** Tier 1 is the company's live leadership
  page or an SEC filing; tier 2 its own press releases; tier 3 news; tier 4
  snippets and directory pages, which are never enough on their own to make
  someone current.
- **It respects refusals.** `robots.txt` and HTTP refusals are honoured, and
  the tool never imitates a browser. A page it may not read is stored as
  `skipped` with the reason. Every run report lists these pages, the
  company's own first.
- **Extraction** (`leadership/extract.py`): an LLM reads each page and
  returns claims as JSON: a role, an appointment or a departure, each with a
  verbatim evidence quote. Before anything is stored, code checks without a
  model that:
  - the quote is on the page;
  - the person's full name is next to it;
  - the title belongs to this person and this company.

  Claims that fail are kept as rejected, with the reason.
- **Verification and enrichment** (`leadership/verify.py`):
  - dates that the held documents state but extraction missed are recovered;
  - one targeted search is made per uncertain role;
  - location and a short biography are looked up for current leaders. They
    are kept only when the quote ties them to the person; otherwise "not
    found" is stored.
- **Bounded and resumable.** Every run has limits on pages, searches, model
  calls and time (`config.toml`). Pages, searches and model calls are
  cached, so a rerun makes no network calls. A company gets one automatic
  search pass; `--refresh` allows another.

### 2. Storage design

SQLite, in two layers (`leadership/db.py`):

- **Claims** are the facts as sources state them. Each one keeps its
  source, evidence quote, and three dates: published, retrieved, and
  effective. The other tables are `sources`, `people` (one row per person,
  merged across name variants), `checks` (verification outcomes) and `runs`
  (status and reasons for each run).
- **Roles** are derived from the claims by fixed rules, never by the model
  (SPEC 3.6, rules 0 to 7):
  - an explicit departure ends a role;
  - an appointment of a successor ends a single-holder role;
  - an announced change waits until it takes effect;
  - an old, uncorroborated role becomes `unverified`;
  - sources that disagree are flagged as a conflict.

  Each role records the rule that decided it. Statuses are `current`,
  `former`, `unverified` and `announced`. Leaving a role is kept apart from
  leaving the company (`employment_status`).
- **Two views are the only thing the chat sees.** `leadership` has every
  role with its source, dates, evidence, conflict, successor and the source
  that ended it. `current_leadership` has current roles only.
- **Dates are honest.** An end date is only ever a date a source states.
  When no date is stated, `ended_on` is empty and `end_reported_on` holds the
  date of the source that reported the end.

Why SQLite rather than a vector store: the questions are structured lookups
and counts over a few hundred rows, which SQL answers exactly. SQL also lets
every answer show the query and the rows behind it.

### 3. Agent architecture

`leadership/agent.py`, `sqlguard.py`, `grounding.py`, `citations.py`:

- **One tool**, `query_db(sql)`, in a loop of at most 5 model calls, with up
  to 2 SQL retries. The model writes SQL, sees the rows, and answers.
- **The SQL guard is enforced by SQLite, not by reading the SQL text.**
  - The agent reads a snapshot that holds only the two views, for the
    selected company only.
  - The snapshot is opened read-only, and an authorizer refuses writes,
    PRAGMA, ATTACH and the schema.
  - One statement per call, at most 50 rows and 5 seconds.
- **Answers only from data.** The prompt gives the schema, examples, the
  company's run status and the collection date. A final answer is accepted
  only after at least one successful query; otherwise the reply is "I could
  not look this up".
- **Claim-level source check** (`grounding.py`): after the model answers,
  code checks each claim against the source that supports it:
  - role held: a source of that person;
  - role ended: the source that showed the end;
  - still at the company: a source of a current role;
  - role change: a source of the old role's end or of the new role.

  It also checks the dates (stated end date, report date, source date), and
  that a planned end date is not stated as a departure. If something fails,
  the model is asked once more with the problems listed. Anything still
  wrong is shown as a warning, never passed silently.
- **Additions made in code, not by the model:**
  - source URLs are appended for named people the model left uncited, from
    the rows it queried;
  - a count answer gets a note when sources could not be read, so "0 found"
    is not read as "none exist".
- **Providers.** Groq first (`openai/gpt-oss-120b`), with OpenAI
  (`gpt-5.4-mini`) as fallback when Groq is rate-limited or unreachable. One
  key is enough.
- **Cached and logged.** Answers are cached by question, prompt, model, data
  and code. Every question, model call, SQL attempt and answer is logged to
  `logs/chat-*.jsonl`.

## Measured results

The full figures are in `acceptance/M4.md` (data) and `acceptance/M5.md`
(chat).

**Data, against hand-verified gold sets** (`fixtures/gold/`):

| | Campfire | Robinhood |
|---|---|---|
| Sources found / read / refused | 25 / 20 / 4 | 73 / 44 / 29 |
| People / stored claims | 5 / 17 | 27 / 149 |
| Precision of `current` roles | 100% | 100% |
| C-level recall | 1 of 1 | **5 of 11** (all 5 whose sources could be read) |
| VP recall | (none in gold set) | **0 of 14** |
| Head recall | 1 of 2 | (none in gold set) |
| Former leaders reported as current | 0 | 0 |

Robinhood's missing leaders are all named only on its leadership page
(`investors.robinhood.com`), which refuses automated clients. The tool does
not get around that. So each recall figure is shown twice: against the whole
gold set, and against the people whose sources could be read (5 of 5).

**Held-out company (Wise, run once before any tuning):** the CEO was found
and correct; 4 people were stored with no errors on a hand-check. Coverage
was 1 of 9, for the same reason: Wise's leadership page refuses the fetcher.
A domain with no leadership information (`example.com`) ends
`insufficient_data` without an error.

**Chat** (`uv run leadership eval --chat`, 31 questions with hand-written
reference SQL, plus 3 special cases):

| Measure | Result |
|---|---|
| Answer correctness | 31 of 31 |
| The four sample questions (CEO, C-level team, C-level count, previous CFO) × 2 companies | 8 of 8 |
| Hallucination (CEO deleted from a test copy), prompt injection (instruction planted in a bio), incomplete data | 3 of 3 |
| Cited URLs that are not a source of anyone the answer names | 0 of 32 |
| Claim-level source check | 34 of 34 (2 after one correction turn) |
| Manual citation review (10 answers, three-question rubric) | 10 of 10 |

## Known limits

- **Coverage is limited by what may be read.** Robinhood's and Wise's
  leadership pages refuse automated clients, so most of their executives are
  missing. Robinhood has no vice presidents in the data. The chat says "not
  found", and count answers say how many sources could not be read.
- **The gold sets are not fully blind.** Claude drafted them from primary
  sources after earlier pipeline runs, and the owner approved them row by
  row.
- **Old roles stay `unverified`.** A 2020 press release does not show
  someone still holds an office.
- **Long documents are read in part.** At most three pieces of an SEC filing
  go to the model.
- **PDFs are found but not read.**
- **No personal location was found** for any leader; "not found" is stored
  and said.
- **The source check finds claims by English keywords** and checks only
  paragraphs about one person. The model is not fully repeatable; the cache
  makes an answer stable, not right.
- **Status rules work from the collection date.** A planned end date that has
  passed, with no later source, is shown as planned, not as confirmed.

The full list is in `SPEC.md` and the "Limitations" section of each
acceptance report.

## How this was built with Claude Code

The whole project was built in Claude Code sessions. The transcripts are in
`session/`. The working loop was:

1. **Spec first.** Claude drafted `SPEC.md` and a requirement checklist
   (`SPEC_checklist.xlsx`, 173 rows). I (the owner) reviewed them, and they
   were revised as decisions were made; the spec is at revision 14.
2. **Milestones on branches.** M0 (setup and key check) through M7
   (delivery) each ran on its own branch. Each ended with:
   - `acceptance/M<n>_results.log`: an automated stage run by
     `tools/acceptance.py`;
   - `acceptance/M<n>.md`: the results, decisions made, issues met, limits,
     and a list of manual tests for me to run.

   A branch was merged only after my sign-off.
3. **I tested by hand and pushed back.** Examples:
   - The gold sets were drafted by Claude and approved row by row.
   - When recall was low, I chose to report it honestly, with two figures,
     rather than tune the gold set.
   - I ruled out getting past bot refusals.
   - In M6 manual testing I found an answer that cited a 2020 press release
     for a 2026 departure. Claude traced it to the model attaching the wrong
     row's source, and built the claim-level source check with a regression
     test from the exact logged answer. Run on the cached evaluation, the
     check then caught two more answers the earlier grader had passed.
   - I pointed out that a planned end date was being stated as a departure;
     that is now checked too.
4. **Real models only, with budgets.** Every model call is real; nothing is
   stubbed. Caching kept reruns free, and each change that needed model
   calls stated its budget first.

To regenerate the export from the local transcripts (the email address and
the `.env` values are redacted):

```bash
uv run python tools/export_session.py
```

## Collecting a new company

```bash
uv run leadership check
```

```bash
uv run leadership collect https://robinhood.com/
```

```bash
uv run leadership export meetcampfire.com robinhood.com
```

- `check` makes a few real calls to confirm the keys and models.
- `collect` runs discovery, fetch, extraction, verification and enrichment
  into `data/leadership.db`. It prints a run report that lists the pages it
  could not read.
- `export` writes `fixtures/<domain>.json` for each company listed. It
  rebuilds `fixtures/leadership.db` with exactly those companies, so list
  every company the chat should keep, plus the new one.

Collection uses optional keys in `.env`:
- `TAVILY_API_KEY` for web search;
- `SEC_USER_AGENT` for SEC EDGAR, which asks automated clients to identify
  themselves.

Without them, those sources are skipped and the run says so. To inspect a
run:

```bash
uv run python tools/show_people.py robinhood.com --checks
```

## Tests and configuration

```bash
uv run pytest
```

runs 385 tests with no network and no keys. `uv run pytest -m llm` runs the
real-model tests. They use the cache where they can, so a rerun makes only a
few calls.

| Command | What it does |
|---|---|
| `uv run leadership eval` | extraction on five hand-labelled pages, and the fixtures against the gold sets (`--gold-only` needs no key) |
| `uv run leadership eval --chat` | the chat evaluation above |
| `uv run python tools/check_evidence.py` | re-checks every stored claim against its saved page |
| `uv run python tools/scan_secrets.py` | scans everything that would be committed for keys and contact addresses |

Configuration:
- `.env` holds secrets only and is never committed.
- `config.toml` holds the models, provider order, collection limits, agent
  limits and evaluation thresholds.

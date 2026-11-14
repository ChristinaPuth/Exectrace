"""Streamlit chat over the collected leadership data. SPEC section 5, "UI".

    uv run python -m streamlit run app.py

Reads the committed fixtures (fixtures/leadership.db); needs one LLM key in
.env and no collection step. Each answer has a panel with the SQL that ran
and the rows it returned.
"""

from __future__ import annotations

import sqlite3

import streamlit as st

from leadership.agent import Agent, collected_on
from leadership.citations import add_missing_sources
from leadership.config import ROOT, ConfigError, configured_providers, load_config, load_env

FIXTURE = ROOT / "fixtures" / "leadership.db"
EXAMPLES = ["Who is the CEO?", "Who are the current C-level executives?",
            "How many C-level executives were found?", "Who was the previous CFO?"]


def companies() -> list[tuple[str, str]]:
    """(domain, name) of the companies in the fixtures, read by this page, not by the model."""
    conn = sqlite3.connect(f"file:{FIXTURE.as_posix()}?mode=ro", uri=True)
    try:
        return conn.execute("SELECT domain, name FROM companies ORDER BY name").fetchall()
    finally:
        conn.close()


@st.cache_resource(show_spinner=False)
def agent_for(domain: str) -> Agent:
    return Agent(load_config(), domain, FIXTURE)


def rows_table(columns: list[str], rows: list[list]) -> str:
    """The rows as a plain Markdown table. Not st.dataframe: that needs pyarrow's compiled libraries,
    which some machines' security policy blocks."""
    def cell(value) -> str:
        text = "" if value is None else str(value)
        text = text if len(text) <= 160 else text[:157] + "..."
        return text.replace("|", "\\|").replace("\n", " ").replace("$", "\\$")

    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def show_answer(entry: dict) -> None:
    text = entry["text"].replace("$", "\\$")           # a dollar sign is not the start of a formula
    if entry["ok"]:
        st.markdown(text)
    else:
        st.error(text)
    with st.expander("SQL and source rows"):
        if not entry["steps"]:
            st.caption("No query ran.")
        for i, step in enumerate(entry["steps"], 1):
            st.code(step.sql or "(empty)", language="sql")
            if step.error:
                st.caption(f"Query {i} was refused or failed: {step.error}")
            elif step.rows:
                st.markdown(rows_table(step.columns, step.rows))
            else:
                st.caption(f"Query {i} returned no rows.")
        st.caption("Answered from the cache." if entry["cached"] else f"Model: {entry['model']}")


def main() -> None:
    st.set_page_config(page_title="Company leadership", page_icon=":material/groups:")
    load_env()
    st.title("Company leadership")
    st.caption("Executive team research from public sources, with the source behind every answer.")
    if not FIXTURE.exists():
        st.error("No data found at fixtures/leadership.db. Run `uv run leadership export` first.")
        return
    options = companies()
    if not options:
        st.error("The database holds no company. Run `uv run leadership collect` and `export` first.")
        return
    with st.sidebar:
        domain = st.selectbox("Company", [d for d, _ in options],
                              format_func=lambda d: dict(options)[d] + f" ({d})")
        st.caption("Answers come only from public sources collected for this company. Each answer has a "
                   "panel with the SQL that ran and the rows behind it.")
        st.markdown("**Try:**\n" + "\n".join(f"- {q}" for q in EXAMPLES))
    if not configured_providers():
        st.error("No LLM key found. Copy .env.example to .env and set GROQ_API_KEY or OPENAI_API_KEY, then "
                 "reload this page.")
        return
    try:
        agent = agent_for(domain)
    except ConfigError as e:
        st.error(f"Configuration problem: {e}")
        return
    with st.container(border=True):
        st.markdown(f"**{agent.company}** &nbsp;·&nbsp; {domain} &nbsp;·&nbsp; "
                    f"Data collected {collected_on(FIXTURE, domain) or 'on an unknown date'} &nbsp;·&nbsp; "
                    f"Collection {agent.status.replace('_', ' ')}")
    if agent.status != "complete":
        st.warning(f"The data collection for this company did not finish normally (status {agent.status}). "
                   "Answers are based on partial data.")
    history: list[dict] = st.session_state.setdefault(f"history:{domain}", [])
    for entry in history:
        with st.chat_message("user", avatar=":material/person:"):
            st.markdown(entry["question"])
        with st.chat_message("assistant", avatar=":material/business_center:"):
            show_answer(entry)
    question = st.chat_input(f"Ask about {agent.company}'s leadership")
    if question:
        with st.chat_message("user", avatar=":material/person:"):
            st.markdown(question)
        with st.chat_message("assistant", avatar=":material/business_center:"):
            with st.spinner("Looking it up in the collected data..."):
                result = agent.ask(question)
            entry = {"question": question, "text": add_missing_sources(result.text, result.steps),
                     "ok": result.ok, "steps": result.steps, "cached": result.cached, "model": result.model}
            show_answer(entry)
        history.append(entry)


main()

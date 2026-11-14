"""Streamlit chat over the collected leadership data. SPEC section 5, "UI".

    uv run python -m streamlit run app.py

Reads the committed fixtures (fixtures/leadership.db); no collection step.
Each answer has a panel with the SQL that ran and the rows it returned.

The LLM key: if the server has one (.env, or the host's secrets), it is used.
If not, as on a public deployment, each visitor enters their own key in the
sidebar and checks it before asking. A visitor's key lives only in that
browser session's memory: it is never put in os.environ (which all visitors
of the server share), never written to disk and never logged.
"""

from __future__ import annotations

import hashlib
import sqlite3

import streamlit as st

from leadership.agent import Agent, collected_on
from leadership.citations import add_count_note, add_missing_sources, unread_sources
from leadership.grounding import grounded_ask
from leadership.config import ROOT, ConfigError, configured_providers, load_config, load_env
from leadership.llm import LLM, check_key

FIXTURE = ROOT / "fixtures" / "leadership.db"
EXAMPLES = ["Who is the CEO?", "Who are the current C-level executives?",
            "How many C-level executives were found?", "Who was the previous CFO?"]
PROVIDER_NAMES = {"groq": "Groq", "openai": "OpenAI"}
KEY_PAGES = {"groq": "https://console.groq.com/keys", "openai": "https://platform.openai.com/api-keys"}


def companies() -> list[tuple[str, str]]:
    """(domain, name) of the companies in the fixtures, read by this page, not by the model."""
    conn = sqlite3.connect(f"file:{FIXTURE.as_posix()}?mode=ro", uri=True)
    try:
        return conn.execute("SELECT domain, name FROM companies ORDER BY name").fetchall()
    finally:
        conn.close()


@st.cache_resource(show_spinner=False)
def agent_for(domain: str) -> Agent:
    """With the server's own key: one agent per company, shared by all sessions."""
    return Agent(load_config(), domain, FIXTURE)


def visitor_agent(domain: str) -> Agent | None:
    """With the visitor's own key: the key box in the sidebar, and an agent for this session only.
    None until a key has been checked and is still the one in the box."""
    cfg = load_config()
    with st.sidebar:
        st.divider()
        st.markdown("**Your API key**")
        provider = st.selectbox("Provider", list(PROVIDER_NAMES), format_func=PROVIDER_NAMES.get, key="provider")
        key = st.text_input(f"{PROVIDER_NAMES[provider]} API key", type="password", key="api_key").strip()
        if st.button("Check key"):
            ok, message = check_key(cfg, provider, key)
            st.session_state["checked"] = _fingerprint(provider, key) if ok else None
            st.session_state["check_message"] = (ok, message)
        current = bool(key) and st.session_state.get("checked") == _fingerprint(provider, key)
        if "check_message" in st.session_state:
            ok, message = st.session_state["check_message"]
            if ok and current:
                st.success(message)
            elif not ok:
                st.error(message)
            else:
                st.info("The key changed. Press Check key again.")
        st.caption(f"The key is used only for your questions and kept only in this browser session's memory; it "
                   f"is not saved or logged. Get one at {KEY_PAGES[provider]}"
                   + (" (free)." if provider == "groq" else "."))
    if not current:
        return None
    agents: dict = st.session_state.setdefault("agents", {})
    name = (domain, _fingerprint(provider, key))
    if name not in agents:
        agents[name] = Agent(cfg, domain, FIXTURE, llm=LLM(cfg, "chat", keys={provider: key}))
    return agents[name]


def _fingerprint(provider: str, key: str) -> str:
    """Identifies a checked key without keeping a second copy of it."""
    return provider + ":" + hashlib.sha256(key.encode()).hexdigest()


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
    try:
        agent = agent_for(domain) if configured_providers() else visitor_agent(domain)
    except ConfigError as e:
        st.error(f"Configuration problem: {e}")
        return
    if agent is None:
        st.info("This app uses your own LLM key. Enter a Groq or OpenAI key in the sidebar and press "
                "**Check key**; then ask about the company's leadership.")
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
                result = grounded_ask(agent, question)
            text = add_missing_sources(result.text, result.steps)
            text = add_count_note(text, result.steps, domain, unread_sources(FIXTURE, domain), agent.status)
            entry = {"question": question, "text": text,
                     "ok": result.ok, "steps": result.steps, "cached": result.cached, "model": result.model}
            show_answer(entry)
        history.append(entry)


main()

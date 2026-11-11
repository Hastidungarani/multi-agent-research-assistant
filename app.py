import time
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agents import critic_chain, llm, writer_chain
from tools import scrape_url, web_search
from utils import (
    as_text, clear_history, load_history, md_to_pdf_bytes, new_entry,
    parse_score, parse_sources, save_run, sources_markdown,
)

# ------------------------------------------------------------------ #
# Page config
# ------------------------------------------------------------------ #
st.set_page_config(page_title="Multi-Agent Research System", page_icon="🔎", layout="wide")
st.title("🔎 Multi-Agent Research System")
st.caption("Search → Reader → Writer → Critic → Revision")

LENGTHS = {
    "Short (~400 words)": "about 400 words",
    "Medium (~800 words)": "about 800 words",
    "Detailed (~1500 words)": "about 1500 words",
}
TONES = ["Professional", "Academic", "Simple and easy to read", "Business brief"]

revise_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are an expert research editor."),
        ("human", """Improve the research report below by fixing every point under
"Areas to Improve" in the critic feedback. Keep the inline citations like [1], [2]
and keep the Sources section. Return ONLY the full revised report.

REPORT:
{report}

CRITIC FEEDBACK:
{feedback}"""),
    ])
    | llm
    | StrOutputParser()
)


def with_retry(fn, tries: int = 3):
    """Run fn(); on rate-limit errors wait for the per-minute limit to reset, then retry."""
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            if "request too large" in msg:
                # A single request bigger than the per-minute limit can never succeed; don't wait.
                raise RuntimeError(
                    "This model's per-minute token limit is too small for this request. "
                    "Switch to a model with a higher limit (set GROQ_MODEL in .env), "
                    "or choose a Short report and turn off auto-revise."
                ) from e
            rate_limited = "429" in msg or "rate_limit" in msg or "rate limit" in msg
            if attempt == tries - 1:
                raise
            wait = 65 if rate_limited else 3
            if rate_limited:
                st.write(f"⏳ Model rate limit reached - waiting {wait}s, then retrying...")
            time.sleep(wait)


# ------------------------------------------------------------------ #
# Sidebar: settings + history
# ------------------------------------------------------------------ #
with st.sidebar:
    st.header("⚙️ Settings")
    n_sources = st.slider("Pages to read", 1, 5, 2)
    length_label = st.selectbox("Report length", list(LENGTHS), index=1)
    tone = st.selectbox("Tone", TONES)
    revise = st.checkbox("Auto-revise if critic score is low", value=True)
    threshold = st.slider("Revise if score below", 1, 10, 7, disabled=not revise)

    st.divider()
    st.header("🕘 History")
    history = load_history()
    if history:
        labels = ["Select a past run..."] + [f"{h['created']} — {h['topic'][:40]}" for h in history]
        choice = st.selectbox("Past runs", labels, label_visibility="collapsed")
        c1, c2 = st.columns(2)
        if c1.button("Open", use_container_width=True) and choice != labels[0]:
            h = history[labels.index(choice) - 1]
            st.session_state.update(state=h["state"], topic=h["topic"], timings=h["timings"])
            st.rerun()
        if c2.button("Clear all", use_container_width=True):
            clear_history()
            st.rerun()
    else:
        st.caption("No saved runs yet.")

    st.divider()
    if st.button("🗑️ Clear current result", use_container_width=True):
        for key in ("state", "topic", "timings"):
            st.session_state.pop(key, None)
        st.rerun()


# ------------------------------------------------------------------ #
# Pipeline
# ------------------------------------------------------------------ #
def run_pipeline_with_ui(topic: str, settings: dict):
    state: dict = {}
    timings: dict = {}

    with st.status("Running research pipeline...", expanded=True) as status:
        # 1) Search
        st.write("🔍 **Step 1** – Searching the web...")
        t = time.time()
        state["search_results"] = with_retry(
            lambda: web_search.invoke(f"{topic} recent reliable detailed information")
        )
        sources = parse_sources(state["search_results"])
        if not sources:
            raise RuntimeError("The search returned no results. Try a different topic.")
        state["sources"] = sources
        timings["Search"] = time.time() - t
        st.write(f"✅ Found {len(sources)} sources ({timings['Search']:.1f}s)")

        # 2) Read
        st.write(f"📖 **Step 2** – Reading the top {settings['n_sources']} page(s)...")
        t = time.time()
        pages = []
        for i, s in enumerate(sources[: settings["n_sources"]], 1):
            content = scrape_url.invoke(s["url"])
            if not content.startswith("Could not scrape"):
                pages.append(f"[{i}] {s['title']} ({s['url']})\n{content}")
        state["scraped_content"] = "\n\n----\n\n".join(pages) or "No page could be scraped."
        timings["Reader"] = time.time() - t
        st.write(f"✅ Read {len(pages)} page(s) ({timings['Reader']:.1f}s)")

        # 3) Write
        st.write("✍️ **Step 3** – Writing the report...")
        t = time.time()
        numbered = "\n".join(
            f"[{i}] {s['title']} - {s['url']}\n    {s['snippet']}" for i, s in enumerate(sources, 1)
        )
        research = (
            f"NUMBERED SOURCES:\n{numbered}\n\n"
            f"DETAILED SCRAPED CONTENT:\n{state['scraped_content']}"
        )
        topic_with_rules = (
            f"{topic}\n\nWriting requirements: length {settings['length']}; tone: {settings['tone']}. "
            "Cite sources inline using their numbers like [1], [2]. "
            "In the Sources section list them as '[n] Title - URL' using the same numbers."
        )
        state["report"] = as_text(
            with_retry(lambda: writer_chain.invoke({"topic": topic_with_rules, "research": research}))
        )
        timings["Writer"] = time.time() - t
        st.write(f"✅ Draft ready ({timings['Writer']:.1f}s)")

        # 4) Critic
        st.write("🧐 **Step 4** – Critic is reviewing...")
        t = time.time()
        state["feedback"] = as_text(with_retry(lambda: critic_chain.invoke({"report": state["report"]})))
        state["score"] = parse_score(state["feedback"])
        timings["Critic"] = time.time() - t
        st.write(f"✅ Review done - score: {state['score'] if state['score'] is not None else 'n/a'}/10 "
                 f"({timings['Critic']:.1f}s)")

        # 5) Revision loop (once)
        state["revised"] = False
        if settings["revise"] and state["score"] is not None and state["score"] < settings["threshold"]:
            st.write("🔁 **Step 5** – Score below threshold, revising the report...")
            t = time.time()
            state["report_v1"], state["feedback_v1"], state["score_v1"] = (
                state["report"], state["feedback"], state["score"],
            )
            state["report"] = as_text(
                with_retry(lambda: revise_chain.invoke(
                    {"report": state["report_v1"], "feedback": state["feedback_v1"]}))
            )
            state["feedback"] = as_text(with_retry(lambda: critic_chain.invoke({"report": state["report"]})))
            state["score"] = parse_score(state["feedback"])
            state["revised"] = True
            timings["Revision"] = time.time() - t
            st.write(f"✅ Revised - new score: {state['score'] if state['score'] is not None else 'n/a'}/10 "
                     f"({timings['Revision']:.1f}s)")

        status.update(label=f"Pipeline complete in {sum(timings.values()):.1f}s",
                      state="complete", expanded=False)

    return state, timings


# ------------------------------------------------------------------ #
# Input
# ------------------------------------------------------------------ #
with st.form("research_form"):
    topic = st.text_input("Research topic", placeholder="e.g. Benefits of solid-state batteries")
    submitted = st.form_submit_button("🚀 Run research", type="primary")

if submitted:
    if not topic.strip():
        st.warning("Please enter a research topic first.")
    else:
        settings = {"n_sources": n_sources, "length": LENGTHS[length_label],
                    "tone": tone, "revise": revise, "threshold": threshold}
        try:
            state, timings = run_pipeline_with_ui(topic.strip(), settings)
            st.session_state.update(state=state, topic=topic.strip(), timings=timings)
            save_run(new_entry(topic.strip(), state, timings, settings))
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.caption("Tip: if this is a rate-limit or 'high demand' error, wait a minute and try again.")

# ------------------------------------------------------------------ #
# Results
# ------------------------------------------------------------------ #
if "state" in st.session_state:
    state = st.session_state["state"]
    topic = st.session_state["topic"]
    timings = st.session_state.get("timings", {})

    st.divider()
    st.subheader(f"Results for: {topic}")

    cols = st.columns(len(timings) + 1)
    for col, (name, secs) in zip(cols, timings.items()):
        col.metric(name, f"{secs:.1f}s")
    score = state.get("score")
    if state.get("revised") and state.get("score_v1") is not None and score is not None:
        cols[-1].metric("Critic score", f"{score:g}/10", f"{score - state['score_v1']:+g} after revision")
    else:
        cols[-1].metric("Critic score", f"{score:g}/10" if score is not None else "n/a")

    tabs = st.tabs(["📄 Report", "🧐 Critic", "📚 Sources", "🔍 Search results", "📖 Scraped content"])

    sources = state.get("sources", [])
    report = state["report"]
    export_md = report
    if sources and "http" not in report:
        export_md += "\n\n## Sources\n" + sources_markdown(sources)

    with tabs[0]:
        st.markdown(report)
        d1, d2, _ = st.columns([1, 1, 3])
        d1.download_button("⬇️ Markdown", data=export_md, file_name="research_report.md",
                           mime="text/markdown", use_container_width=True)
        pdf = md_to_pdf_bytes(topic, export_md)
        if pdf:
            d2.download_button("⬇️ PDF", data=pdf, file_name="research_report.pdf",
                               mime="application/pdf", use_container_width=True)
        else:
            d2.caption("Install `reportlab` for PDF export.")
        if state.get("revised"):
            with st.expander("See the first draft (before revision)"):
                st.markdown(state["report_v1"])
                st.caption(f"First draft score: {state['score_v1']:g}/10")
    with tabs[1]:
        st.markdown(state["feedback"])
        if state.get("revised"):
            with st.expander("Feedback on the first draft"):
                st.markdown(state["feedback_v1"])
    with tabs[2]:
        st.markdown(sources_markdown(sources) if sources else "No sources.")
    with tabs[3]:
        st.text(state["search_results"])
    with tabs[4]:
        st.text(state["scraped_content"])
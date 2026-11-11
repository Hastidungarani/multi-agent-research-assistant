"""Helper functions for the research app (no Streamlit / LLM imports, easy to test)."""
import io
import json
import re
from datetime import datetime
from pathlib import Path
HISTORY_FILE = Path(__file__).parent / "history.json"
MAX_HISTORY = 30
def as_text(value) -> str:
    """Normalise str / message / list-of-blocks to plain text and drop <think> blocks."""
    if hasattr(value, "content"):
        value = value.content
    if isinstance(value, list):
        value = "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in value
        )
    return re.sub(r"<think>.*?</think>", "", str(value), flags=re.DOTALL).strip()
def parse_sources(search_text: str) -> list[dict]:
    """Parse the output of web_search into [{title, url, snippet}, ...]."""
    pattern = r"Title:\s*(.*?)\nURL:\s*(\S+)\nSnippet:\s*(.*?)(?=\n----|\Z)"
    return [
        {"title": t.strip(), "url": u.strip(), "snippet": s.strip()}
        for t, u, s in re.findall(pattern, search_text, flags=re.DOTALL)
    ]
def parse_score(feedback: str):
    """Extract the 'Score: X/10' value from the critic's feedback, or None."""
    m = re.search(r"Score:\s*\**\s*(\d+(?:\.\d+)?)\s*/\s*10", feedback or "", re.I)
    return float(m.group(1)) if m else None
def sources_markdown(sources: list[dict]) -> str:
    return "\n".join(
        f"{i}. [{s['title']}]({s['url']})" for i, s in enumerate(sources, 1)
    )
def load_history() -> list[dict]:
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
def save_run(entry: dict) -> None:
    history = load_history()
    history.insert(0, entry)
    HISTORY_FILE.write_text(
        json.dumps(history[:MAX_HISTORY], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
def clear_history() -> None:
    if HISTORY_FILE.exists():
        HISTORY_FILE.unlink()
def new_entry(topic: str, state: dict, timings: dict, settings: dict) -> dict:
    return {
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "topic": topic,
        "settings": settings,
        "state": state,
        "timings": timings,
    }
_REPLACEMENTS = {
    "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u2022": "-", "\u00a0": " ",
}
def _clean(text: str) -> str:
    for a, b in _REPLACEMENTS.items():
        text = text.replace(a, b)
    return text.encode("latin-1", "replace").decode("latin-1")
def _inline(text: str) -> str:
    """Convert a little inline markdown to reportlab markup."""
    text = _clean(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)",
                  r'<link href="\2" color="blue">\1</link>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return text
def md_to_pdf_bytes(title: str, markdown: str):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    except ImportError:
        return None
    styles = getSampleStyleSheet()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title=_clean(title),
                            leftMargin=54, rightMargin=54, topMargin=54, bottomMargin=54)
    story = [Paragraph(_inline(title), styles["Title"]), Spacer(1, 12)]

    for raw in markdown.splitlines():
        line = raw.rstrip()
        if not line.strip():
            story.append(Spacer(1, 6))
        elif line.startswith("### "):
            story.append(Paragraph(_inline(line[4:]), styles["Heading3"]))
        elif line.startswith("## "):
            story.append(Paragraph(_inline(line[3:]), styles["Heading2"]))
        elif line.startswith("# "):
            story.append(Paragraph(_inline(line[2:]), styles["Heading1"]))
        elif re.match(r"^\s*[-*]\s+", line):
            body = re.sub(r"^\s*[-*]\s+", "", line)
            story.append(Paragraph(_inline(body), styles["BodyText"],
                                   bulletText="\u2022"))
        else:
            story.append(Paragraph(_inline(line), styles["BodyText"]))

    doc.build(story)
    return buf.getvalue()
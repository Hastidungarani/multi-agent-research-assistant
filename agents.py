from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_url
import os
from dotenv import load_dotenv
load_dotenv()
PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
def _supports_tools(model_name: str) -> bool:
    """Try a tiny tool-enabled call. Only a clear 'no tool support / no such model' counts as False."""
    from langchain_groq import ChatGroq
    try:
        ChatGroq(model=model_name, temperature=0, max_retries=0, timeout=20) \
            .bind_tools([web_search]).invoke("Say hi")
        return True
    except Exception as e:
        msg = str(e).lower()
        if "tool" in msg or "model_not_found" in msg or "does not exist" in msg or "decommission" in msg:
            return False
        return True  # e.g. rate limit: the model itself is fine
def pick_groq_model() -> str:
    """Use GROQ_MODEL from .env if set, otherwise find a chat model that supports tool calling."""
    chosen = os.getenv("GROQ_MODEL")
    if chosen:
        return chosen
    from groq import Groq
    models = [m.id for m in Groq(api_key=os.getenv("GROQ_API_KEY")).models.list().data]
    skip = ("whisper", "guard", "tts", "playai", "orpheus", "compound", "safeguard", "gpt-oss", "qwen")
    all_chat = [m for m in models if not any(x in m.lower() for x in ("whisper", "guard", "tts", "playai", "orpheus", "compound", "safeguard"))]
    models = [m for m in models if not any(x in m.lower() for x in skip)]
    def rank(name):
        n = name.lower()
        for i, kw in enumerate(("llama-3.3-70b", "70b", "llama-3.1-8b", "instant", "llama")):
            if kw in n:
                return i
        return 99
    models.sort(key=rank)
    for m in models[:8]:
        if _supports_tools(m):
            return m
    if all_chat:
        return all_chat[0]
    raise RuntimeError("No usable Groq chat model found for your key.")
def get_llm():
    if PROVIDER == "groq":
        from langchain_groq import ChatGroq

        model_name = pick_groq_model()
        print(f"[agents] Using Groq model: {model_name}")
        return ChatGroq(model=model_name, temperature=0, timeout=90, max_retries=1,
                        max_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", "2000")))

    if PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
            temperature=0,
            timeout=60,
            max_retries=1,
        )

    if PROVIDER == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.1"), temperature=0)

    if PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model="gpt-4o-mini", temperature=0)

    raise ValueError(f"Unknown LLM_PROVIDER: {PROVIDER}")

llm = get_llm()
def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search]
    )
def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url]
    )
writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. writer clear, structured and insightful reports."),
    ("human", """ write a detailed research report on the topic below.
Topic: {topic}
research gathered: {research}
structure the report as:
- Introduction
- key findings(minimum 3 well-explained points)
- Conclusion
- sources (list all URLs found in the research)
Be detailed, factual and professional.""")
])

writer_chain = writer_prompt | llm | StrOutputParser()
critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.
Report:
{report}
Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])
critic_chain = critic_prompt | llm | StrOutputParser()
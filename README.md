# multi-agent-research-assistant
Multi-agent LLM research pipeline: live web search, cited reports, and a critic that triggers automatic revision. Python, LangChain, Streamlit.

Turn any topic into a sourced research report. The app searches the web, reads the best pages, writes a report with inline citations, has a critic score it, and automatically revises it when the score is low.

Built with Python, LangChain, Streamlit, Tavily and Groq.

![App demo](docs/demo.png).

## How it works

1. Search    Tavily finds recent, relevant sources
2. Reader    BeautifulSoup scrapes the top pages
3. Writer    LLM drafts a structured report with [1], [2] citations
4. Critic    LLM scores the report (X/10) with strengths and improvements
5. Revision  If the score is below your threshold, the writer revises
             the report using the critic's feedback, then it is scored again

## Features

- Cited reports: inline citations like [1], [2] and a clickable Sources tab
- Critic loop: automatic revision when the score is low, with a before/after score comparison
- Adjustable settings: pages to read, report length, tone and revision threshold
- Live progress: step-by-step status with per-step timing
- Run history: past reports are saved locally and can be reopened from the sidebar
- Export: download reports as Markdown or PDF
- Provider-agnostic: switch between Groq, Gemini, OpenAI and Ollama with one line in .env
- Resilient: retries on temporary API errors and waits out rate limits

## Tech stack

- Interface - Streamlit
- LLM orchestration	- LangChain (prompt templates, chains)
- LLM providers -	Groq (default), Google Gemini, Ollama
- Web search - Tavily API
- Scraping - Requests, BeautifulSoup
- PDF export - ReportLab

## Getting started

1. Clone and install

   git clone https://github.com/Hastidungarani/multi-agent-research-assistant.git
   
       cd multi-agent-research-assistant
   
       python -m venv .venv
       source .venv/bin/activate
       pip install -r requirements.txt
  
3. Add your API keys

       cp .env.example .env

Then edit .env:

      LLM_PROVIDER=groq
      GROQ_API_KEY=your_groq_key
      GROQ_MODEL=your_model_name
      MAX_OUTPUT_TOKENS=2000
      TAVILY_API_KEY=your_tavily_key

      Free Groq key: console.groq.com
      Free Tavily key: tavily.com

3. Run

       python -m streamlit run app.py
   
## Project structure

- ├── app.py            Streamlit UI and pipeline orchestration
- ├── agents.py         LLM provider setup, writer and critic chains
- ├── tools.py          Tavily search and page scraping tools
- ├── utils.py          Parsing, run history and PDF export helpers
- ├── requirements.txt
- ├── .env.example      Template for your own keys
- └── README.md

## Limitations

- Report quality depends on the pages that can be scraped. Some sites block scrapers.
- The critic is an LLM, so its scores are a useful signal but not ground truth.
- Live data topics (weather, scores, prices) are a poor fit for a report-style pipeline.
- Free LLM tiers have rate limits, which makes long runs slower.

## Roadmap

- Read pages in parallel to cut run time
- Rank sources by quality and recency
- Add a benchmark script that measures speed and critic scores across topics
- Deploy a hosted demo

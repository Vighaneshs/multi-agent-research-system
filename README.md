# Multi-Agent Research System

LangGraph-based system that coordinates four specialised agents to automate research tasks.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph Orchestrator                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Planner ──▶ Search ──▶ Code ──▶ Synthesis                 │
│                                                             │
│  Breaks the     Tavily +   Sandboxed      Combines          │
│  query into     arXiv      Python exec    everything         │
│  subtasks       lookups                   into a report      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Setup

```bash
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# fill in GOOGLE_API_KEY and TAVILY_API_KEY
```

Get your keys:
- **Google API key**: https://aistudio.google.com/apikey
- **Tavily API key**: https://app.tavily.com

## Usage

```bash
# single query
python main.py "What are the latest advances in protein folding?"

# interactive mode
python main.py
```

## Project layout

```
├── main.py                 # entry point
├── config.py               # settings from .env
├── agents/
│   ├── base_agent.py       # shared LLM + retry logic
│   ├── planner_agent.py    # decomposes query into tasks
│   ├── search_agent.py     # web + arxiv search
│   ├── code_agent.py       # sandboxed code execution
│   └── synthesis_agent.py  # final report generation
├── graph/
│   ├── state.py            # TypedDict state definition
│   └── workflow.py         # LangGraph nodes & edges
├── tools/
│   ├── web_search.py       # Tavily / DuckDuckGo wrappers
│   ├── arxiv_search.py     # arXiv API wrapper
│   └── code_executor.py    # restricted exec() tool
└── tests/
    └── test_agents.py
```

## Running tests

```bash
pytest tests/ -v
```

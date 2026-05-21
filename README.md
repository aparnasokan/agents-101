# Agent 101 — Live Demo System

A fully working multi-agent AI system built for live presentations.
Run the Python backend, open the browser, and demo AI agents *using* AI agents.

## Quickstart

```bash
# 1. Clone / copy this folder

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Azure OpenAI settings
cp .env.example .env
# Edit .env with your Azure OpenAI endpoint, deployment, and API key

# 4. Run
python main.py
# Browser opens automatically at http://localhost:8000
```

Required `.env` values:

```env
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=your-deployment-name
AZURE_OPENAI_API_VERSION=2024-10-21
```

## Architecture

```
agent101/
├── main.py                  # FastAPI server — entry point
├── memory.py                # Shared session state
├── agents/
│   ├── base.py              # Base class (Azure OpenAI API call)
│   ├── orchestrator.py      # Routes messages to specialist agents
│   ├── guide.py             # Narrative & demo transitions
│   ├── concept.py           # Explains agent concepts
│   ├── viz.py               # SVG diagrams (fires with concept)
│   ├── code_agent.py        # Python code patterns
│   ├── best_practices.py    # Dos, don'ts, gotchas
│   ├── mcp_agent.py         # Model Context Protocol
│   ├── deploy.py            # Production & hosting
│   ├── qa.py                # Live audience questions
│   └── scribe.py            # End-of-demo cheat sheet
├── static/
│   └── index.html           # The UI (served by FastAPI)
├── requirements.txt
└── .env.example
```

## Standalone HTML Demo

`agent101-demo.html` is a separate single-file demo that calls Azure OpenAI directly from the browser.
Before using it, update the Azure constants near the bottom of that file with your endpoint, deployment, API version, and key.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Serves the UI |
| POST | `/api/chat` | Main pipeline — orchestrates and runs agents |
| POST | `/api/scribe` | Generates cheat sheet from session history |
| GET | `/api/stats` | Session stats |
| POST | `/api/clear` | Clears session memory |

## Demo Flow

1. **Intro** — Guide agent sets the scene
2. **Concepts** — Concept explains first, then Visual renders a supporting diagram
3. **Code** — Code agent shows implementation patterns
4. **Best Practices** — Best Practices agent covers dos/don'ts
5. **MCP** — MCP agent explains Model Context Protocol
6. **Deploy** — Deploy agent covers production patterns
7. **Q&A** — Q&A agent handles live audience questions
8. **Wrap-up** — Hit "Generate Cheat Sheet" for the Scribe agent

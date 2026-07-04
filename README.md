# Intent-Driven Agentic Platform

Give it any topic. It figures out what it actually needs to know (asking
only for what's missing), plans a small pipeline of agent steps, and
executes them for real — no mocked output.

## Design constraints

- **No external LLM API calls.** All inference runs against a locally
  hosted model via [Ollama](https://ollama.com) (`OLLAMA_HOST`, default
  `http://localhost:11434`). Nothing in this repo calls Anthropic, OpenAI,
  or any other hosted LLM API.
- **No keyed search/data APIs.** Real-world data comes from direct HTTP
  requests + HTML parsing: `app/tools/web_search.py` scrapes DuckDuckGo's
  plain HTML results page, `app/tools/web_fetch.py` fetches and cleans the
  text of a given URL. No API keys anywhere.
- Storage is SQLite by default (`DATABASE_URL` env var), swappable to
  Postgres by changing the connection string — see `docker-compose.yml`.

## How it works

1. `POST /session {topic}` — the topic is classified into an intent, and
   the model generates a schema of **mandatory** and **optional** slots
   specific to that topic (e.g. "plan a garden" → mandatory: hardiness
   zone; optional: aesthetic style, budget).
2. If mandatory slots are missing, the response comes back with
   `status: "awaiting_input"` and a list of clarifying `questions`. Answer
   them via `POST /session/{id}/answer {slot_values}`. Optional slots get
   sensible defaults automatically.
3. Once all mandatory slots are filled, the model produces an ordered plan
   (a small DAG of steps using the tools `web_search`, `web_fetch`,
   `summarize`, `synthesize`).
4. The plan executes for real: it searches the live web, fetches and reads
   actual pages, and synthesizes a final answer — all traced in the
   `plan_steps` table for inspection.

This is all orchestrated by a LangGraph `StateGraph` in `backend/app/graph.py`.
Each of the early nodes (`classify_intent`, `generate_schema`) is a no-op if
its output is already present in the carried-forward state, which is what
lets the same compiled graph safely resume after a clarification round
without needing a LangGraph checkpointer — session state round-trips
through the `sessions.graph_state` JSON column between HTTP requests.

## Setup

```bash
# 1. Install and run a local model
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2
ollama serve &   # if not already running as a service

# 2. Install Python deps
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Run the app
uvicorn app.main:app --reload
```

Open `http://localhost:8000`.

Or run the whole stack (app + Ollama) with Docker:

```bash
docker compose up --build
ollama pull llama3.2   # or: docker compose exec ollama ollama pull llama3.2
```

## Tests

```bash
cd backend && pip install -r requirements.txt   # if not already done
cd ..
pytest tests/ -v
```

Tests mock the LLM and web calls, so they run without Ollama or internet
access — CI runs the same suite on every push (`.github/workflows/ci.yml`).

## Pushing to GitHub

This was scaffolded on a machine with no outbound network access to
GitHub. To publish it:

```bash
gh repo create <your-repo-name> --private --source=. --remote=origin --push
```

or, if you created the repo in the GitHub UI first:

```bash
git remote add origin <your-repo-url>
git push -u origin main
```

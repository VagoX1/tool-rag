# Tool-enabled local RAG starter

A complete agentic RAG starter that keeps source documents, vectors, and conversation history local. OpenAI provides embeddings and reasoning; an allowlisted backend executes retrieval and external tools.

> The included `company-handbook.md` is fictional demonstration data.

## Features

- Hybrid retrieval combining semantic similarity with BM25-style keyword ranking.
- Incremental ingestion that embeds only new or changed chunks.
- Markdown, text, and PDF document support.
- LangChain document objects and recursive text splitting.
- Optional LangChain-native, document-only retrieval-to-answer chain.
- Exact source allowlists enforced by the backend, not the model.
- Persistent multi-turn sessions in local SQLite.
- Citation extraction and warnings for missing or invented citations.
- Direct search without paying for answer generation.
- Index statistics and document inventory.
- Interactive CLI chat and a FastAPI HTTP service.
- Live, read-only weather integration through Open-Meteo.
- Optional hosted web search with clickable URL citations and low-context defaults.
- Safe local math for arithmetic, algebra, calculus, statistics, matrices, and units.
- Strict tool schemas, fixed network hosts, timeouts, and bounded tool rounds.

## Architecture

```text
Documents -> LangChain splitting -> OpenAI embeddings -> local hybrid index
                                                           |
Question  -> retrieval -> agent and allowlisted tools -> cited answer
```

The index combines semantic vector similarity with BM25-style keyword ranking.
The standard agent can select local retrieval, web search, weather, or math tools.
The separate LangChain chain answers strictly from retrieved document context.

## Requirements

- Python 3.12 (tested version)
- An OpenAI API account with billing configured
- An API key with Request access for Responses and Embeddings
- Web Search Request access if you want to use web search

## Set up

PowerShell:

```powershell
git clone https://github.com/YOUR-USERNAME/tool-rag-starter.git
cd tool-rag-starter
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:OPENAI_API_KEY="your-key"
```

Your restricted OpenAI key needs `Request` access for Responses and Embeddings. Enable `Web Search: Request` if it is shown separately in your key settings. Never put the key in a source file or commit it.

Each person running a cloned copy supplies their own API key and pays for their
own API usage. If you deploy this project as a hosted service, keep the key only
in the host's server-side secret storage; usage by your visitors is then billed
to your OpenAI account.

## Add and ingest documents

Place `.md`, `.txt`, or `.pdf` files in `documents/`, then run:

```powershell
python -m app.cli ingest
python -m app.cli status
```

Ingestion reports `embedded` and `reused` chunk counts. Running it again without changing documents reuses every existing vector and makes no embedding request.

## Search and ask

Search the index directly:

```powershell
python -m app.cli search "credential exposure" --top-k 3
```

Ask the agent:

```powershell
python -m app.cli ask "What is the hardware return window?"
python -m app.cli ask "What is the current weather in Berlin?"
```

Use the simpler LangChain pipeline when you want an answer strictly from indexed
documents, without the agent's web, math, weather, or conversation tools:

```powershell
python -m app.cli langchain-ask "What is the hardware return window?"
python -m app.cli langchain-ask "Summarize the return rules" `
  --source company-handbook.md --top-k 5
```

The normal `ask` command remains the tool-enabled agent. The `langchain-ask`
command uses LangChain prompt/model/output runnables and returns validated local
citations.

Restrict retrieval to one or more exact source names:

```powershell
python -m app.cli ask "Summarize the return rules" `
  --source company-handbook.md
```

## Conversation sessions

Start an interactive conversation with a newly generated session ID:

```powershell
python -m app.cli chat
```

Resume or share a named session:

```powershell
python -m app.cli chat --session support-demo
python -m app.cli ask "What timezone is that?" --session support-demo
```

Inspect or delete local history:

```powershell
python -m app.cli sessions
python -m app.cli clear-session support-demo
```

Only user and assistant messages are persisted. Tool-call internals and model reasoning are not stored in the session database.

## Web search

The model searches the web only for recent, current, or explicitly requested public information. Search context defaults to `low`, and simple lookups are instructed to use at most one search.

```powershell
python -m app.cli ask "Search the web for today's major AI news"
```

Disable web access for a request or entire chat:

```powershell
python -m app.cli ask "Answer only from my documents" --no-web
python -m app.cli chat --no-web
```

Web search and model pricing can change. Check the
[current OpenAI API pricing](https://developers.openai.com/api/docs/pricing)
before enabling it for other users. Cited URLs are returned in `web_sources`;
the answer also contains inline links.

## Math tools

Math is calculated locally with SymPy and Python's statistics library, so it has no separate tool-call fee. The model uses verified tool results to explain the answer.

```powershell
python -m app.cli ask "What is 17.5% of 480?"
python -m app.cli ask "Solve x^2 - 5x + 6 = 0"
python -m app.cli ask "Differentiate sin(x) * x^2"
python -m app.cli ask "Find the inverse of [[1,2],[3,4]]"
python -m app.cli ask "Convert 72 Fahrenheit to Celsius"
```

The parser allows numbers, variables, arithmetic operators, powers, parentheses, and an allowlist of common functions. It rejects imports, attributes, indexing, arbitrary function calls, oversized expressions, and excessive exponents. It never uses Python `eval()`.

## HTTP API

Start the service:

```powershell
python -m app.cli serve
```

> This starter API does not authenticate callers. Do not expose it directly to
> the public internet without adding authentication, rate limits, request-size
> limits, usage monitoring, and a spending cap.

Open `http://127.0.0.1:8000/docs`, or ask a question:

```powershell
$body = @{
  question = "When is customer support available?"
  sources = @("company-handbook.md")
  allow_web = $false
} | ConvertTo-Json

$response = Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/ask `
  -ContentType application/json `
  -Body $body

$response.session_id
```

Send the returned `session_id` with the next `/ask` request to continue the conversation.

Endpoints:

- `GET /health`
- `GET /index/status`
- `GET /documents`
- `GET /sessions`
- `DELETE /sessions/{session_id}`
- `POST /ingest`
- `POST /search` with `query`, optional `top_k`, and optional `sources`
- `POST /ask` with `question`, optional `session_id`, optional `sources`, and `allow_web`
- `POST /langchain/ask` with `question`, optional `sources`, and optional `top_k`
  for document-only RAG

## Storage and safety

The model receives tool schemas but never credentials or executable Python access. The backend owns the tool allowlist and source filter. Retrieved text and tool output are explicitly treated as untrusted data. Responses use `store=False`.

The Git-ignored files `data/index.json` and `data/sessions.db` contain private document or conversation data. Protect and back them up according to the sensitivity of the source material.

Before adding a tool that writes data, such as sending mail or modifying a database, add server-side user authorization, argument validation, an idempotency key, an audit record, and explicit user approval immediately before execution.

## Configuration

- `RAG_CHAT_MODEL`: defaults to `gpt-5.6-luna`.
- `RAG_EMBEDDING_MODEL`: defaults to `text-embedding-3-small`.
- `RAG_REASONING_EFFORT`: defaults to `low`.
- `RAG_TOP_K`: defaults to `4`.
- `RAG_MAX_TOOL_ROUNDS`: defaults to `4`.
- `RAG_HISTORY_MESSAGES`: defaults to the most recent `12` messages.
- `RAG_WEB_SEARCH_ENABLED`: defaults to `true`; set `false` to disable it globally.
- `RAG_WEB_SEARCH_CONTEXT`: defaults to `low`; supported values are `low`, `medium`, and `high`.

Changing the embedding model requires ingestion again. Lowering history length reduces multi-turn token usage.

## Tests

```powershell
pytest
```

## License

Released under the [MIT License](LICENSE).

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .agent import RAGAgent
from .config import Settings, require_api_key
from .langchain_rag import LangChainRAG


app = FastAPI(title="Tool-enabled RAG", version="2.0.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10_000)
    session_id: str | None = Field(default=None, max_length=100)
    sources: list[str] | None = Field(default=None, max_length=50)
    allow_web: bool = True


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    top_k: int = Field(default=4, ge=1, le=20)
    sources: list[str] | None = Field(default=None, max_length=50)


class LangChainAskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10_000)
    sources: list[str] | None = Field(default=None, max_length=50)
    top_k: int | None = Field(default=None, ge=1, le=20)


class IngestResponse(BaseModel):
    documents: int
    chunks: int
    embedded: int
    reused: int


@lru_cache(maxsize=1)
def get_agent() -> RAGAgent:
    require_api_key()
    return RAGAgent(Settings())


@app.get("/health")
def health() -> dict[str, object]:
    settings = Settings()
    return {
        "ok": True,
        "index_exists": settings.index_path.exists(),
        "model": settings.chat_model,
        "embedding_model": settings.embedding_model,
    }


@app.get("/index/status")
def index_status() -> dict[str, object]:
    try:
        return get_agent().store.status()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/documents")
def documents() -> list[dict[str, object]]:
    try:
        return get_agent().store.documents()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/sessions")
def sessions() -> list[dict[str, object]]:
    try:
        return get_agent().sessions.sessions()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> dict[str, object]:
    try:
        deleted_messages = get_agent().sessions.clear(session_id)
        return {"session_id": session_id, "deleted_messages": deleted_messages}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ingest", response_model=IngestResponse)
def ingest() -> dict[str, int]:
    try:
        return get_agent().store.ingest()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/search")
def search(payload: SearchRequest) -> dict[str, object]:
    try:
        results = get_agent().store.search(
            query=payload.query,
            top_k=payload.top_k,
            sources=payload.sources,
        )
        return {"query": payload.query, "results": results}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ask")
def ask(payload: AskRequest) -> dict[str, object]:
    try:
        agent = get_agent()
        session_id = payload.session_id or agent.sessions.new_id()
        result = agent.ask(
            payload.question,
            session_id=session_id,
            sources=payload.sources,
            allow_web=payload.allow_web,
        )
        return {
            "answer": result.answer,
            "session_id": result.session_id,
            "citations": result.citations,
            "web_sources": result.web_sources,
            "citation_warnings": result.citation_warnings,
            "tool_calls": result.tool_calls,
            "usage": result.usage,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/langchain/ask")
def langchain_ask(payload: LangChainAskRequest) -> dict[str, object]:
    """Direct document-only RAG using a LangChain prompt/model pipeline."""
    try:
        agent = get_agent()
        result = LangChainRAG(agent.store, agent.settings).ask(
            payload.question,
            sources=payload.sources,
            top_k=payload.top_k,
        )
        return {
            "answer": result.answer,
            "citations": result.citations,
            "citation_warnings": result.citation_warnings,
            "retrieved_chunks": result.retrieved_chunks,
            "framework": "langchain",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    documents_dir: Path = ROOT / "documents"
    index_path: Path = ROOT / "data" / "index.json"
    sessions_path: Path = ROOT / "data" / "sessions.db"
    chat_model: str = os.getenv("RAG_CHAT_MODEL", "gpt-5.6-luna")
    embedding_model: str = os.getenv(
        "RAG_EMBEDDING_MODEL", "text-embedding-3-small"
    )
    reasoning_effort: str = os.getenv("RAG_REASONING_EFFORT", "low")
    top_k: int = int(os.getenv("RAG_TOP_K", "4"))
    max_tool_rounds: int = int(os.getenv("RAG_MAX_TOOL_ROUNDS", "4"))
    history_messages: int = int(os.getenv("RAG_HISTORY_MESSAGES", "12"))
    web_search_enabled: bool = env_bool("RAG_WEB_SEARCH_ENABLED", True)
    web_search_context: str = os.getenv("RAG_WEB_SEARCH_CONTEXT", "low")


def require_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Set it in your environment before running."
        )

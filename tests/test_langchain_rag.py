from __future__ import annotations

from pathlib import Path

from langchain_core.runnables import RunnableLambda

from app.config import Settings
from app.langchain_documents import load_and_split_documents
from app.langchain_rag import LangChainRAG


class FakeStore:
    def search(self, query, top_k=None, sources=None):
        return [
            {
                "citation": "guide.md#chunk-1",
                "source": "guide.md",
                "text": "Returns are accepted for 30 days.",
            }
        ]


def test_langchain_loader_splits_text_and_preserves_source(tmp_path: Path) -> None:
    (tmp_path / "guide.md").write_text("Alpha. " * 100, encoding="utf-8")
    paths, documents = load_and_split_documents(
        tmp_path, chunk_size=100, chunk_overlap=10
    )
    assert len(paths) == 1
    assert len(documents) > 1
    assert {document.metadata["source"] for document in documents} == {"guide.md"}


def test_langchain_rag_validates_citations(tmp_path: Path) -> None:
    model = RunnableLambda(
        lambda prompt: "The return period is 30 days [guide.md#chunk-1]."
    )
    rag = LangChainRAG(
        FakeStore(),
        Settings(index_path=tmp_path / "index.json"),
        chat_model=model,
    )
    result = rag.ask("What is the return period?")
    assert result.retrieved_chunks == 1
    assert result.citations[0]["citation"] == "guide.md#chunk-1"
    assert result.citation_warnings == []

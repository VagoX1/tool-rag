from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.agent import extract_web_sources, validate_citations
from app.config import Settings
from app.retrieval import Chunk, LocalVectorStore
from app.sessions import SessionStore
from app.tools import (
    convert_units,
    evaluate_math,
    matrix_math,
    parse_math,
    symbolic_math,
)


class FakeEmbeddings:
    def __init__(self) -> None:
        self.inputs: list[object] = []

    def create(self, *, input: object, **_: object) -> SimpleNamespace:
        self.inputs.append(input)
        values = input if isinstance(input, list) else [input]
        data = [SimpleNamespace(embedding=[1.0, 0.0]) for _ in values]
        return SimpleNamespace(data=data)


class FakeClient:
    def __init__(self) -> None:
        self.embeddings = FakeEmbeddings()


def settings_for(tmp_path: Path) -> Settings:
    return Settings(
        documents_dir=tmp_path / "documents",
        index_path=tmp_path / "data" / "index.json",
        sessions_path=tmp_path / "data" / "sessions.db",
    )


def test_incremental_ingest_reuses_unchanged_embeddings(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    settings.documents_dir.mkdir()
    (settings.documents_dir / "guide.md").write_text(
        "The launch code is ZXQ-77.", encoding="utf-8"
    )
    client = FakeClient()
    store = LocalVectorStore(settings, client)  # type: ignore[arg-type]

    first = store.ingest()
    second = store.ingest()

    assert first["embedded"] == 1
    assert second["embedded"] == 0
    assert second["reused"] == 1
    assert len(client.embeddings.inputs) == 1


def test_hybrid_search_promotes_exact_terms_and_filters_sources(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    client = FakeClient()
    store = LocalVectorStore(settings, client)  # type: ignore[arg-type]
    store.chunks = [
        Chunk("general.md#chunk-1", "general.md", "A general launch guide.", [1.0, 0.0]),
        Chunk("codes.md#chunk-1", "codes.md", "Launch code ZXQ-77.", [1.0, 0.0]),
    ]

    results = store.search("ZXQ-77", top_k=2)
    filtered = store.search("launch", top_k=2, sources=["general.md"])

    assert results[0]["source"] == "codes.md"
    assert [item["source"] for item in filtered] == ["general.md"]


def test_sessions_persist_and_limit_history(tmp_path: Path) -> None:
    sessions = SessionStore(tmp_path / "sessions.db")
    sessions.append_exchange("abc", "first", "one")
    sessions.append_exchange("abc", "second", "two")

    assert sessions.history("abc", limit=2) == [
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "two"},
    ]
    assert sessions.sessions()[0]["messages"] == 4
    assert sessions.clear("abc") == 4
    assert sessions.history("abc") == []


def test_citation_validation_reports_missing_and_invalid_citations() -> None:
    available = {
        "guide.md#chunk-1": {
            "citation": "guide.md#chunk-1",
            "source": "guide.md",
        }
    }
    citations, warnings = validate_citations(
        "Valid [guide.md#chunk-1], invalid [fake.md#chunk-9].",
        available,
        used_search=True,
    )

    assert citations == [available["guide.md#chunk-1"]]
    assert "fake.md#chunk-9" in warnings[0]


def test_safe_math_evaluates_and_blocks_code() -> None:
    assert evaluate_math("(15/100) * 240")["exact"] == "36"
    assert symbolic_math("solve", "x^2 - 5x + 6 = 0", "x")["solutions"] == [
        "2",
        "3",
    ]
    try:
        parse_math("__import__('os').system('whoami')")
    except ValueError as exc:
        assert "allowed" in str(exc) or "Unsupported" in str(exc)
    else:
        raise AssertionError("Unsafe expression was accepted")


def test_symbolic_matrix_and_unit_math() -> None:
    assert symbolic_math("solve", "x^2 - 4 = 0", "x")["solutions"] == ["-2", "2"]
    assert matrix_math("determinant", [[1, 2], [3, 4]], None)["result"] == "-2"
    assert convert_units(1, "km", "m")["value"] == 1000
    assert round(convert_units(32, "F", "C")["value"], 8) == 0


def test_web_source_extraction_deduplicates_annotations() -> None:
    annotation = SimpleNamespace(
        type="url_citation", url="https://example.com/a", title="Example"
    )
    message = SimpleNamespace(
        type="message",
        content=[SimpleNamespace(annotations=[annotation, annotation])],
    )
    assert extract_web_sources([message]) == [
        {"title": "Example", "url": "https://example.com/a"}
    ]

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from openai import OpenAI
from pypdf import PdfReader

from .config import Settings
from .langchain_documents import load_and_split_documents


TOKEN_PATTERN = re.compile(r"[\w'-]+", re.UNICODE)


@dataclass
class Chunk:
    id: str
    source: str
    text: str
    embedding: list[float]
    content_hash: str = ""


def normalize(vector: Sequence[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        return [0.0 for _ in vector]
    return [value / magnitude for value in vector]


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def chunk_text(text: str, max_chars: int = 1400, overlap_chars: int = 200) -> list[str]:
    """Split text on paragraph boundaries with a small character overlap."""
    clean = re.sub(r"\r\n?", "\n", text)
    clean = re.sub(r"[ \t]+", " ", clean)
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", clean) if part.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        pieces = (
            [paragraph]
            if len(paragraph) <= max_chars
            else [
                paragraph[start : start + max_chars]
                for start in range(0, len(paragraph), max_chars - overlap_chars)
            ]
        )
        for piece in pieces:
            candidate = f"{current}\n\n{piece}".strip() if current else piece
            if current and len(candidate) > max_chars:
                chunks.append(current)
                prefix = current[-overlap_chars:].lstrip()
                current = f"{prefix}\n\n{piece}".strip()
                if len(current) > max_chars:
                    chunks.append(current[:max_chars])
                    current = current[max_chars - overlap_chars :]
            else:
                current = candidate

    if current:
        chunks.append(current)
    return chunks


def read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8")


def batched(values: Sequence[str], size: int = 64) -> Iterable[Sequence[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class LocalVectorStore:
    def __init__(self, settings: Settings, client: OpenAI | None = None):
        self.settings = settings
        self.client = client or OpenAI()
        self.chunks: list[Chunk] = []
        self._load_if_present()

    def _load_if_present(self) -> None:
        if not self.settings.index_path.exists():
            return
        payload = json.loads(self.settings.index_path.read_text(encoding="utf-8"))
        indexed_model = payload.get("embedding_model")
        if indexed_model != self.settings.embedding_model:
            raise RuntimeError(
                f"Index uses {indexed_model}, configured model is "
                f"{self.settings.embedding_model}. Run ingestion again."
            )
        self.chunks = [Chunk(**item) for item in payload.get("chunks", [])]
        for chunk in self.chunks:
            if not chunk.content_hash:
                chunk.content_hash = _hash_text(chunk.text)

    def ingest(self) -> dict[str, int]:
        paths, documents = load_and_split_documents(self.settings.documents_dir)
        pending: list[tuple[str, str, str, str]] = []
        chunk_numbers: Counter[str] = Counter()
        for document in documents:
            source = str(document.metadata["source"])
            text = document.page_content.strip()
            if not text:
                continue
            chunk_numbers[source] += 1
            number = chunk_numbers[source]
            pending.append(
                (f"{source}#chunk-{number}", source, text, _hash_text(text))
            )

        if not pending:
            raise RuntimeError(
                f"No .md, .txt, or .pdf documents found in {self.settings.documents_dir}."
            )

        previous_vectors = {
            chunk.content_hash: chunk.embedding
            for chunk in self.chunks
            if chunk.content_hash and chunk.embedding
        }
        missing: dict[str, str] = {
            content_hash: text
            for _, _, text, content_hash in pending
            if content_hash not in previous_vectors
        }
        new_vectors: dict[str, list[float]] = {}
        missing_items = list(missing.items())
        for batch_start in range(0, len(missing_items), 64):
            batch_items = missing_items[batch_start : batch_start + 64]
            response = self.client.embeddings.create(
                model=self.settings.embedding_model,
                input=[text for _, text in batch_items],
                encoding_format="float",
            )
            for (content_hash, _), item in zip(batch_items, response.data, strict=True):
                new_vectors[content_hash] = normalize(item.embedding)

        vectors = previous_vectors | new_vectors
        self.chunks = [
            Chunk(
                id=chunk_id,
                source=source,
                text=text,
                embedding=vectors[content_hash],
                content_hash=content_hash,
            )
            for chunk_id, source, text, content_hash in pending
        ]
        payload = {
            "version": 2,
            "embedding_model": self.settings.embedding_model,
            "chunks": [asdict(chunk) for chunk in self.chunks],
        }
        self.settings.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings.index_path.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        reused = sum(1 for *_, content_hash in pending if content_hash in previous_vectors)
        return {
            "documents": len(paths),
            "chunks": len(self.chunks),
            "embedded": len(missing),
            "reused": reused,
        }

    def _filtered_chunks(self, sources: list[str] | None) -> list[Chunk]:
        if not sources:
            return self.chunks
        allowed = set(sources)
        return [chunk for chunk in self.chunks if chunk.source in allowed]

    def search(
        self,
        query: str,
        top_k: int | None = None,
        sources: list[str] | None = None,
    ) -> list[dict[str, object]]:
        candidates = self._filtered_chunks(sources)
        if not self.chunks:
            raise RuntimeError("The knowledge index is empty. Run ingestion first.")
        if not candidates:
            return []

        response = self.client.embeddings.create(
            model=self.settings.embedding_model,
            input=query,
            encoding_format="float",
        )
        query_vector = normalize(response.data[0].embedding)
        semantic_scores = {
            chunk.id: sum(a * b for a, b in zip(query_vector, chunk.embedding))
            for chunk in candidates
        }

        query_terms = tokenize(query)
        term_frequencies = {chunk.id: Counter(tokenize(chunk.text)) for chunk in candidates}
        document_frequencies = Counter(
            term for term in set(query_terms) for counts in term_frequencies.values() if term in counts
        )
        average_length = sum(sum(counts.values()) for counts in term_frequencies.values()) / len(candidates)
        lexical_scores: dict[str, float] = {}
        for chunk in candidates:
            counts = term_frequencies[chunk.id]
            length = max(1, sum(counts.values()))
            score = 0.0
            for term in query_terms:
                frequency = counts[term]
                if not frequency:
                    continue
                inverse_frequency = math.log(
                    1 + (len(candidates) - document_frequencies[term] + 0.5)
                    / (document_frequencies[term] + 0.5)
                )
                score += inverse_frequency * (
                    frequency * 2.2
                    / (frequency + 1.2 * (0.25 + 0.75 * length / average_length))
                )
            lexical_scores[chunk.id] = score

        semantic_rank = {
            chunk_id: rank
            for rank, (chunk_id, _) in enumerate(
                sorted(semantic_scores.items(), key=lambda item: item[1], reverse=True),
                start=1,
            )
        }
        lexical_rank = {
            chunk_id: rank
            for rank, (chunk_id, _) in enumerate(
                sorted(lexical_scores.items(), key=lambda item: item[1], reverse=True),
                start=1,
            )
        }
        combined = {
            chunk.id: 0.45 / (60 + semantic_rank[chunk.id])
            + 0.55 / (60 + lexical_rank[chunk.id])
            for chunk in candidates
        }
        ranked = sorted(candidates, key=lambda chunk: combined[chunk.id], reverse=True)
        limit = top_k or self.settings.top_k
        return [
            {
                "citation": chunk.id,
                "source": chunk.source,
                "score": round(combined[chunk.id], 6),
                "semantic_score": round(semantic_scores[chunk.id], 4),
                "lexical_score": round(lexical_scores[chunk.id], 4),
                "text": chunk.text,
            }
            for chunk in ranked[:limit]
        ]

    def documents(self) -> list[dict[str, object]]:
        counts = Counter(chunk.source for chunk in self.chunks)
        return [
            {"source": source, "chunks": chunks}
            for source, chunks in sorted(counts.items())
        ]

    def status(self) -> dict[str, object]:
        return {
            "index_exists": self.settings.index_path.exists(),
            "embedding_model": self.settings.embedding_model,
            "documents": len({chunk.source for chunk in self.chunks}),
            "chunks": len(self.chunks),
            "sources": [item["source"] for item in self.documents()],
        }

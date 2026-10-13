from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}


def document_paths(documents_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in documents_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_and_split_documents(
    documents_dir: Path,
    chunk_size: int = 1400,
    chunk_overlap: int = 200,
) -> tuple[list[Path], list[Document]]:
    """Load supported files and split them into LangChain Documents."""
    paths = document_paths(documents_dir)
    loaded: list[Document] = []

    for path in paths:
        source = path.relative_to(documents_dir).as_posix()
        if path.suffix.lower() == ".pdf":
            reader = PdfReader(str(path))
            loaded.extend(
                Document(
                    page_content=page.extract_text() or "",
                    metadata={"source": source, "page": page_number},
                )
                for page_number, page in enumerate(reader.pages)
            )
        else:
            loaded.append(
                Document(
                    page_content=path.read_text(encoding="utf-8"),
                    metadata={"source": source},
                )
            )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return paths, splitter.split_documents(loaded)

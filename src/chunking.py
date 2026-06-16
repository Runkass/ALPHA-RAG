"""Split documents into overlapping chunks."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import get_settings


@dataclass(frozen=True)
class Chunk:
    chunk_id: int
    web_id: int
    url: str
    title: str
    chunk_index: int
    text: str


def split_document(web_id: int, url: str, title: str, text: str) -> list[Chunk]:
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    pieces = splitter.split_text(text)
    chunks: list[Chunk] = []
    for i, piece in enumerate(pieces):
        piece = piece.strip()
        if not piece:
            continue
        chunks.append(
            Chunk(
                chunk_id=-1,
                web_id=web_id,
                url=url,
                title=title,
                chunk_index=i,
                text=piece,
            )
        )
    return chunks

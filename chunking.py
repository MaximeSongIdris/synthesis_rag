"""Chunk DoclingDocuments into aligned lists of texts and sources."""

from dataclasses import dataclass
from pathlib import Path

from docling.chunking import HybridChunker
from docling_core.types.doc import DoclingDocument


@dataclass
class ChunkSource:
    document: str
    headings: list[str]
    pages: list[int]

def chunk_documents(json_dir: str | Path, chunker: HybridChunker,
                    min_words: int = 20) -> tuple[list[str], list[str], list[ChunkSource]]:
    """Restore every DoclingDocument saved as JSON in `json_dir`,
    then chunk them into three lists: texts, texts_without_title and sources."""
    texts: list[str] = []                # document title + headings + content -> LLM summaries and search index
    texts_without_title: list[str] = []  # headings + content -> ideal for clustering task
    sources: list[ChunkSource] = []      # where each chunk comes from

    seen: set[str] = set()
    for json_path in sorted(Path(json_dir).rglob("*.json", case_sensitive=False)):
        doc = DoclingDocument.load_from_json(json_path)  # restore the DoclingDocument from its saved JSON
        for chunk in chunker.chunk(dl_doc=doc):
            if len(chunk.text.split()) < min_words or chunk.text in seen:
                continue  # noise or exact duplicate
            seen.add(chunk.text)
            text_without_title = chunker.contextualize(chunk=chunk)  # section headings + chunk content
            texts.append(f"{doc.name}\n{text_without_title}")
            texts_without_title.append(text_without_title)
            pages = sorted({p.page_no for item in chunk.meta.doc_items for p in item.prov})
            sources.append(ChunkSource(doc.name, chunk.meta.headings or [], pages))
    return texts, texts_without_title, sources
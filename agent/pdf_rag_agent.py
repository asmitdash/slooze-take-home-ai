"""Challenge B — PDF RAG Agent.

Pipeline:
  1. Extract text from PDF (pypdf, page-aware)
  2. Chunk by ~800-char windows with overlap
  3. Embed chunks via sentence-transformers (all-MiniLM-L6-v2, 384-d, local)
  4. Index with FAISS (in-memory cosine via inner product on L2-normalized vectors)
  5. For a question: embed query -> retrieve top-k chunks -> Claude synthesises
     a grounded answer

Trade-offs:
  - Local embeddings (MiniLM) instead of Bedrock embeddings keeps the demo
    self-contained and free; quality is sufficient for document QA at this
    scale.
  - FAISS in-memory because the challenge expects a single document; for
    multi-document or persisted use, swap in Chroma with the same interface.
  - Chunking is character-window based rather than sentence-aware; simple
    and predictable. A semantic chunker would help on heterogeneous PDFs.
  - Summary path retrieves a wider set of chunks (top-12) so the LLM can
    actually see the whole document, not just the densely-similar ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from .llm import BedrockClaude


CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_QA = 5
TOP_K_SUMMARY = 12


@dataclass
class Chunk:
    text: str
    page: int
    chunk_id: int


SUMMARY_SYSTEM = """You are a precise document summariser.
Given excerpts from a single PDF, produce a faithful summary covering the document's purpose, main claims, and any methodology or results mentioned. Stay grounded — if the excerpts don't cover something, omit it rather than guess."""


QA_SYSTEM = """You answer questions about a single document using only the supplied excerpts.
- Quote or paraphrase tightly from the excerpts.
- If the excerpts do not contain the answer, say so explicitly.
- Always end with a "Sources" line listing the excerpt numbers you used (e.g., [1], [3])."""


def _extract_pages(pdf_path: Path) -> List[Tuple[int, str]]:
    reader = PdfReader(str(pdf_path))
    out = []
    for i, page in enumerate(reader.pages, 1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            out.append((i, text))
    return out


def _chunk(pages: List[Tuple[int, str]]) -> List[Chunk]:
    chunks: List[Chunk] = []
    cid = 0
    for page_num, text in pages:
        text = " ".join(text.split())
        start = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            piece = text[start:end].strip()
            if piece:
                chunks.append(Chunk(text=piece, page=page_num, chunk_id=cid))
                cid += 1
            if end == len(text):
                break
            start = end - CHUNK_OVERLAP
    return chunks


class PdfRag:
    def __init__(self, pdf_path: str | Path):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(self.pdf_path)
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.llm = BedrockClaude()
        self.chunks: List[Chunk] = []
        self.index: faiss.Index | None = None
        self._build()

    def _build(self) -> None:
        pages = _extract_pages(self.pdf_path)
        if not pages:
            raise ValueError(f"No extractable text in {self.pdf_path}")
        self.chunks = _chunk(pages)
        if not self.chunks:
            raise ValueError("No chunks produced from PDF")
        vecs = self.embedder.encode(
            [c.text for c in self.chunks],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        self.index = index

    def _retrieve(self, query: str, k: int) -> List[Chunk]:
        qv = self.embedder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        scores, idxs = self.index.search(qv, min(k, len(self.chunks)))
        return [self.chunks[i] for i in idxs[0] if i != -1]

    def _format_context(self, retrieved: List[Chunk]) -> str:
        return "\n\n".join(
            f"[{n}] (page {c.page})\n{c.text}" for n, c in enumerate(retrieved, 1)
        )

    def summarize(self) -> str:
        retrieved = self.chunks[: TOP_K_SUMMARY] if len(self.chunks) <= TOP_K_SUMMARY else self._spread_sample(TOP_K_SUMMARY)
        context = self._format_context(retrieved)
        prompt = f"Document excerpts:\n{context}\n\nWrite the summary now."
        return self.llm.complete(system=SUMMARY_SYSTEM, user=prompt, max_tokens=900).text

    def _spread_sample(self, k: int) -> List[Chunk]:
        if len(self.chunks) <= k:
            return self.chunks
        step = len(self.chunks) / k
        return [self.chunks[int(i * step)] for i in range(k)]

    def ask(self, question: str, k: int = TOP_K_QA) -> str:
        retrieved = self._retrieve(question, k=k)
        context = self._format_context(retrieved)
        prompt = (
            f"Question: {question}\n\nDocument excerpts:\n{context}\n\nAnswer the question now."
        )
        return self.llm.complete(system=QA_SYSTEM, user=prompt, max_tokens=900).text

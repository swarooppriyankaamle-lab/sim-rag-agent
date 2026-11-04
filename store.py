"""Document loading, chunking, embedding and a small in-memory vector store."""
from __future__ import annotations

import pickle
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

SUPPORTED = {".md", ".txt", ".pdf"}
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


@dataclass
class Chunk:
    id: str        # "<filename>#<n>", e.g. "contact_convergence.md#2"
    source: str    # file name
    section: str   # nearest markdown heading
    text: str
    index: int     # position within its source document


# ---------------------------------------------------------------- loading

def load_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return "\n\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown into (heading, body) pairs; plain text becomes one section."""
    matches = list(HEADING_RE.finditer(text))
    if not matches:
        return [("", text)]
    sections = []
    if text[: matches[0].start()].strip():
        sections.append(("", text[: matches[0].start()]))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((m.group(2).strip(), text[m.end():end]))
    return [(h, b) for h, b in sections if b.strip()]


def window(text: str, size: int, overlap: int) -> list[str]:
    """Character windows that break on whitespace, with overlap."""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= size:
        return [text] if text else []
    pieces, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(" ", start + size // 2, end)
            if cut > start:
                end = cut
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
        nxt = text.find(" ", start)
        if 0 <= nxt < end:
            start = nxt + 1
    return [p for p in pieces if p]


def chunk_documents(docs_dir: Path, size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.rglob("*")):
        if path.suffix.lower() not in SUPPORTED or not path.is_file():
            continue
        n = 0
        for heading, body in split_sections(load_text(path)):
            for piece in window(body, size, overlap):
                chunks.append(Chunk(f"{path.name}#{n}", path.name, heading, piece, n))
                n += 1
    return chunks


# ---------------------------------------------------------------- embedders

class SentenceTransformerEmbedder:
    backend = "st"

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def fit(self, texts):  # pretrained, nothing to fit
        pass

    def encode(self, texts) -> np.ndarray:
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True), dtype=np.float32)


class TfidfEmbedder:
    """Offline fallback. Lexical, not semantic, but fine for small corpora and testing."""
    backend = "tfidf"

    def __init__(self, vectorizer=None):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = vectorizer or TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")

    def fit(self, texts):
        self.vectorizer.fit(list(texts))

    def encode(self, texts) -> np.ndarray:
        m = self.vectorizer.transform(list(texts)).toarray().astype(np.float32)
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.where(norms == 0, 1, norms)


def make_embedder(backend: str, model_name: str):
    if backend == "tfidf":
        return TfidfEmbedder()
    if backend == "st":
        return SentenceTransformerEmbedder(model_name)
    try:
        return SentenceTransformerEmbedder(model_name)
    except Exception as exc:  # missing package or no network to download the model
        print(f"[store] sentence-transformers unavailable ({type(exc).__name__}); using TF-IDF.", file=sys.stderr)
        return TfidfEmbedder()


def embed_text(c: Chunk) -> str:
    return f"{c.section}\n{c.text}" if c.section else c.text


# ---------------------------------------------------------------- store

class VectorStore:
    def __init__(self, chunks: list[Chunk], embedder, matrix: np.ndarray):
        self.chunks = chunks
        self.embedder = embedder
        self.matrix = matrix
        self._by_id = {c.id: c for c in chunks}

    @classmethod
    def build(cls, chunks: list[Chunk], backend: str, model_name: str) -> "VectorStore":
        if not chunks:
            raise ValueError("No chunks to index. Add .md, .txt or .pdf files to docs/.")
        embedder = make_embedder(backend, model_name)
        texts = [embed_text(c) for c in chunks]
        embedder.fit(texts)
        return cls(chunks, embedder, embedder.encode(texts))

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        q = self.embedder.encode([query])[0]
        scores = self.matrix @ q
        order = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in order]

    def get(self, chunk_id: str) -> Chunk | None:
        return self._by_id.get(chunk_id)

    def neighbors(self, chunk_id: str, radius: int = 1) -> list[Chunk]:
        c = self.get(chunk_id)
        if c is None:
            return []
        return [x for x in self.chunks if x.source == c.source and abs(x.index - c.index) <= radius]

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "chunks": [asdict(c) for c in self.chunks],
            "matrix": self.matrix,
            "backend": self.embedder.backend,
            "model_name": getattr(self.embedder, "model_name", None),
            "vectorizer": getattr(self.embedder, "vectorizer", None),
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        if not path.exists():
            raise FileNotFoundError(f"No index at {path}. Run: python ingest.py")
        with open(path, "rb") as f:
            state = pickle.load(f)
        if state["backend"] == "tfidf":
            embedder = TfidfEmbedder(state["vectorizer"])
        else:
            embedder = SentenceTransformerEmbedder(state["model_name"])
        return cls([Chunk(**c) for c in state["chunks"]], embedder, state["matrix"])

"""
ingest.py
---------
Creates and searches the FAISS RAG index for the DHA Construction & Development
Byelaws PDF.

One-way dependency:
    app.py -> ingest.py
    ingest.py does NOT import app.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


INDEX_DIR = Path("faiss_index")
INDEX_FILE = INDEX_DIR / "byelaws.faiss"
META_FILE = INDEX_DIR / "metadata.json"

# Small local embedding model. No embedding API key is required.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_chunks(text: str, max_chars: int = 1400, overlap: int = 220) -> List[str]:
    """
    Splits page text into reasonably sized semantic-ish chunks.
    We keep page boundaries because page number is important evidence.
    """
    text = clean_text(text)
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            # Hard split an unusually long paragraph.
            if current:
                chunks.append(current)
                current = ""
            start = 0
            while start < len(paragraph):
                end = min(start + max_chars, len(paragraph))
                piece = paragraph[start:end].strip()
                if piece:
                    chunks.append(piece)
                start = max(end - overlap, start + 1)
            continue

        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            tail = current[-overlap:] if current else ""
            current = f"{tail}\n\n{paragraph}".strip()

    if current:
        chunks.append(current)

    return chunks


def detect_heading(text: str) -> str:
    """
    Attempts to capture the first visible numbered section/heading.
    This is metadata only; the original byelaw text remains authoritative.
    """
    patterns = [
        r"(?im)^\s*(SECTION\s+\d+[^.\n]*)",
        r"(?im)^\s*(\d+\.\s+[A-Z][A-Z0-9 &()/,\-]{4,})",
        r"(?im)^\s*(\d+\s+[A-Z][A-Z0-9 &()/,\-]{4,})",
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip()
    return ""


def load_byelaws(pdf_path: str | Path) -> List[Dict[str, Any]]:
    pdf_path = Path(pdf_path)
    reader = PdfReader(str(pdf_path))
    records: List[Dict[str, Any]] = []

    for page_index, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        page_text = clean_text(raw)
        if not page_text:
            continue

        heading = detect_heading(page_text)
        chunks = split_into_chunks(page_text)

        for chunk_no, chunk in enumerate(chunks, start=1):
            records.append(
                {
                    "id": f"P{page_index:02d}-C{chunk_no:02d}",
                    "page": page_index,
                    "section": heading,
                    "text": chunk,
                    "source": pdf_path.name,
                }
            )

    return records


def build_index(pdf_path: str | Path, force: bool = False) -> Dict[str, Any]:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    if INDEX_FILE.exists() and META_FILE.exists() and not force:
        return {
            "status": "exists",
            "index": str(INDEX_FILE),
            "metadata": str(META_FILE),
        }

    records = load_byelaws(pdf_path)
    if not records:
        raise ValueError("No text could be extracted from the Byelaws PDF.")

    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [
        f"Section: {r['section']}\nPage: {r['page']}\n{r['text']}"
        for r in records
    ]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_FILE))
    META_FILE.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "status": "built",
        "chunks": len(records),
        "dimension": int(embeddings.shape[1]),
        "index": str(INDEX_FILE),
        "metadata": str(META_FILE),
    }


def load_index():
    if not INDEX_FILE.exists() or not META_FILE.exists():
        raise FileNotFoundError(
            "FAISS index not found. Run build_index() first."
        )

    index = faiss.read_index(str(INDEX_FILE))
    records = json.loads(META_FILE.read_text(encoding="utf-8"))
    model = SentenceTransformer(EMBEDDING_MODEL)
    return index, records, model


def search_byelaws(query: str, top_k: int = 8) -> List[Dict[str, Any]]:
    index, records, model = load_index()

    vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, ids = index.search(vector, min(top_k, len(records)))

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        item = dict(records[int(idx)])
        item["score"] = float(score)
        results.append(item)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build DHA Byelaws FAISS index")
    parser.add_argument("pdf", help="Path to DHA Byelaws PDF")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the index even if it already exists",
    )
    args = parser.parse_args()

    result = build_index(args.pdf, force=args.force)
    print(json.dumps(result, indent=2))

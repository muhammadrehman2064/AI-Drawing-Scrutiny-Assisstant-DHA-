"""
ingest.py
---------
Loads and searches the PRE-BUILT DHA Byelaws FAISS RAG index.

IMPORTANT:
- This file does NOT build/rebuild the Byelaws index at runtime.
- The FAISS index and metadata must already exist.
- The index should be generated once in Google Colab.
- Streamlit only loads the existing index and performs query embedding.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# PRE-BUILT INDEX FILES
# These files are already present in the GitHub repository.
# ============================================================

INDEX_FILE = Path("byelaws.faiss")
META_FILE = Path("metadata.json")

# Same embedding model that was used to create the FAISS index
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# LOAD PRE-BUILT KNOWLEDGE BASE
# ============================================================

def load_index():
    """
    Load the already-created FAISS index and metadata.

    NO PDF processing.
    NO chunking.
    NO re-embedding of Byelaws.
    NO index creation.
    """

    if not INDEX_FILE.exists():
        raise FileNotFoundError(
            f"Pre-built FAISS index not found: {INDEX_FILE}"
        )

    if not META_FILE.exists():
        raise FileNotFoundError(
            f"Pre-built metadata file not found: {META_FILE}"
        )

    # Load existing FAISS index
    index = faiss.read_index(str(INDEX_FILE))

    # Load existing metadata
    records = json.loads(
        META_FILE.read_text(encoding="utf-8")
    )

    # Load embedding model ONLY for embedding the user's query
    model = SentenceTransformer(EMBEDDING_MODEL)

    return index, records, model


# ============================================================
# SEARCH EXISTING BYELAWS INDEX
# ============================================================

def search_byelaws(
    query: str,
    top_k: int = 8
) -> List[Dict[str, Any]]:
    """
    Search the pre-built FAISS Byelaws index.

    The Byelaws themselves are NOT embedded again.
    Only the user's query is converted into an embedding.
    """

    if not query or not query.strip():
        return []

    index, records, model = load_index()

    # Embed ONLY the user's query
    vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    # Search existing FAISS vectors
    k = min(top_k, index.ntotal)

    if k <= 0:
        return []

    scores, ids = index.search(vector, k)

    results = []

    for score, idx in zip(scores[0], ids[0]):

        if idx < 0:
            continue

        idx = int(idx)

        if idx >= len(records):
            continue

        item = dict(records[idx])

        item["score"] = float(score)

        results.append(item)

    return results

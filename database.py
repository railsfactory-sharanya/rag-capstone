"""Shared Qdrant client and embedding model.

Same role as the engine in the tutorial's database.py: created once when
this module is imported, then handed to each request. MiniLM is the
expensive object, so a request must not construct it again.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from ingest import (
    COLLECTION_NAME,
    MODEL_NAME,
    VECTOR_SIZE,
    ensure_category_index,
    get_client,
)


class AppResources:
    """The objects a route needs. One instance lives for the process."""

    def __init__(self, qdrant, embedder: SentenceTransformer):
        self.qdrant = qdrant
        self.embedder = embedder

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.asarray(
            self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False),
            dtype=np.float32,
        )
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape != (len(texts), VECTOR_SIZE):
            raise ValueError(f"Expected vectors of shape {(len(texts), VECTOR_SIZE)}, got {vectors.shape}")
        return vectors


def _load_resources() -> AppResources:
    qdrant = get_client()
    if not qdrant.collection_exists(COLLECTION_NAME):
        raise SystemExit(f"Collection {COLLECTION_NAME} is missing. Run: python ingest.py --fresh")
    ensure_category_index(qdrant)
    print(f"Loading {MODEL_NAME}")
    embedder = SentenceTransformer(MODEL_NAME, device="cpu")
    return AppResources(qdrant, embedder)


resources = _load_resources()


def get_resources():
    """Yield the process-wide client and model.

    The tutorial's get_db opens a new session per request and closes it
    after the response. A session holds transaction state. This client
    and this model do not, and building the model per request is the
    cost generate.py pays on every command.
    """
    yield resources

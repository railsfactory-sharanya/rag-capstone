"""Plain similarity search against the mealbox collection.

Embeds one question with the same MiniLM model used at ingest, then asks
Qdrant for the nearest 3 points. No category filter and no reranker.

    python search.py
    python search.py "How do I pause for a month?"
"""

from __future__ import annotations

import argparse

import numpy as np
from sentence_transformers import SentenceTransformer

from ingest import COLLECTION_NAME, MODEL_NAME, ensure_category_index, get_client

DEFAULT_QUERY = "How do I pause for a month?"
TOP_K = 3
TEXT_PREVIEW = 180


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one raw similarity search on mealbox.")
    parser.add_argument(
        "query",
        nargs="?",
        default=DEFAULT_QUERY,
        help=f"Question to embed. Default: {DEFAULT_QUERY}",
    )
    return parser.parse_args()


def embed_query(query: str) -> np.ndarray:
    print(f"Loading {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    vector = model.encode(query, convert_to_numpy=True).astype(np.float32)
    if vector.shape != (384,):
        raise ValueError(f"Expected a 384-d query vector, got {vector.shape}")
    print(f"query vector: shape={vector.shape} dtype={vector.dtype} norm={float(np.linalg.norm(vector)):.4f}")
    return vector


def preview(text: str) -> str:
    flat = " ".join(text.split())
    if len(flat) <= TEXT_PREVIEW:
        return flat
    return flat[: TEXT_PREVIEW - 3] + "..."


def main() -> None:
    args = parse_args()
    query = args.query.strip()
    if not query:
        raise SystemExit("Query is empty.")

    client = get_client()
    if not client.collection_exists(COLLECTION_NAME):
        raise SystemExit(f"Collection {COLLECTION_NAME} is missing. Run: python ingest.py --fresh")
    ensure_category_index(client)

    print(f"query: {query}")
    vector = embed_query(query)
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector.tolist(),
        limit=TOP_K,
        with_payload=True,
    )

    if not response.points:
        raise SystemExit("Search returned no points.")

    for rank, point in enumerate(response.points, start=1):
        payload = point.payload or {}
        print(
            f"{rank}. score={point.score:.4f} point_id={point.id} "
            f"entry_id={payload.get('entry_id')} kind={payload.get('kind')} "
            f"category={payload.get('category')} last_updated={payload.get('last_updated')}"
        )
        print(f"   title: {payload.get('title')}")
        print(f"   text: {preview(str(payload.get('text', '')))}")


if __name__ == "__main__":
    main()

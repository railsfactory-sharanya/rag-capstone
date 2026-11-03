"""Retrieval on top of plain similarity search.

Optional category filter, a title-overlap rerank, and one query rewrite.
The measured before/after is in retrieval-notes.md.

    python retrieve.py
    python retrieve.py --category subscription
    python retrieve.py --category billing "I was charged twice for the same week"
"""

from __future__ import annotations

import argparse
import re

import numpy as np
from qdrant_client import models
from sentence_transformers import SentenceTransformer

from ingest import CATEGORIES, COLLECTION_NAME, MODEL_NAME, ensure_category_index, get_client
from search import preview

DEFAULT_QUERY = "I am traveling for a month and do not want boxes"
TOP_K = 3
CANDIDATE_K = 10
TITLE_WEIGHT = 0.15

# Tiny closed list. "boxes" is intentionally not here: that word is why the
# plain search misses, and stripping it would hide the mistake.
STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "i",
    "my",
    "the",
    "to",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter, rewrite, and rerank a Mealbox search.")
    parser.add_argument(
        "query",
        nargs="?",
        default=DEFAULT_QUERY,
        help=f"Customer question. Default: {DEFAULT_QUERY}",
    )
    parser.add_argument(
        "--category",
        choices=sorted(CATEGORIES),
        help="Keep only points in this category. Uses the keyword payload index.",
    )
    return parser.parse_args()


def rewrite_query(query: str) -> str:
    """Add the pause policy words when a trip is worded as not wanting boxes.

    Returns the original string when the rule does not apply.
    """
    lowered = query.lower()
    away = any(term in lowered for term in ("travel", "traveling", "travelling", "vacation"))
    span = any(term in lowered for term in ("month", "weeks"))
    if away and span:
        return f"{query}. pause the weekly plan for several weeks while away"
    return query


def tokens(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-z0-9]+", text.lower()) if tok not in STOPWORDS}


def title_overlap(query: str, title: str) -> float:
    query_tokens = tokens(query)
    if not query_tokens:
        return 0.0
    return len(query_tokens & tokens(title)) / len(query_tokens)


def category_filter(category: str | None) -> models.Filter | None:
    if category is None:
        return None
    return models.Filter(
        must=[models.FieldCondition(key="category", match=models.MatchValue(value=category))]
    )


def embed(model: SentenceTransformer, text: str) -> np.ndarray:
    vector = model.encode(text, convert_to_numpy=True).astype(np.float32)
    if vector.shape != (384,):
        raise ValueError(f"Expected a 384-d query vector, got {vector.shape}")
    return vector


def search(client, vector: np.ndarray, limit: int, query_filter: models.Filter | None):
    return client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector.tolist(),
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
    ).points


def rerank(query: str, points: list) -> list[tuple[float, float, object]]:
    """Combine cosine with the share of query tokens that appear in the title."""
    ranked = []
    for point in points:
        title = str((point.payload or {}).get("title", ""))
        overlap = title_overlap(query, title)
        combined = float(point.score) + TITLE_WEIGHT * overlap
        ranked.append((combined, overlap, point))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked


def print_plain(points: list) -> None:
    if not points:
        print("   no points")
        return
    for rank, point in enumerate(points, start=1):
        payload = point.payload or {}
        print(
            f"{rank}. score={point.score:.4f} "
            f"entry_id={payload.get('entry_id')} category={payload.get('category')}"
        )
        print(f"   title: {payload.get('title')}")
        print(f"   text: {preview(str(payload.get('text', '')))}")


def print_reranked(rows: list[tuple[float, float, object]]) -> None:
    if not rows:
        print("   no points")
        return
    for rank, (combined, overlap, point) in enumerate(rows, start=1):
        payload = point.payload or {}
        print(
            f"{rank}. rerank={combined:.4f} cosine={point.score:.4f} "
            f"title_overlap={overlap:.2f} "
            f"entry_id={payload.get('entry_id')} category={payload.get('category')}"
        )
        print(f"   title: {payload.get('title')}")
        print(f"   text: {preview(str(payload.get('text', '')))}")


def retrieve_ranked(client, model: SentenceTransformer, query: str, category: str | None = None):
    """Return the raw top 3 and the rewritten, filtered, reranked top 3.

    `before` ignores category so a caller can still show the plain baseline.
    `after` is what generate.py reads.
    """
    rewritten = rewrite_query(query)
    plain_vector = embed(model, query)
    rewritten_vector = plain_vector if rewritten == query else embed(model, rewritten)
    before = search(client, plain_vector, TOP_K, None)
    after = rerank(rewritten, search(client, rewritten_vector, CANDIDATE_K, category_filter(category)))[:TOP_K]
    return rewritten, plain_vector, before, after


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
    if args.category:
        print(f"category filter: {args.category}")
    else:
        print("category filter: none")

    print(f"Loading {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    rewritten, plain_vector, before, after = retrieve_ranked(client, model, query, args.category)
    if rewritten == query:
        print("rewrite: unchanged")
    else:
        print(f"rewrite: {rewritten}")
    print(f"query vector norm={float(np.linalg.norm(plain_vector)):.4f}")

    print(f"\nbefore: plain top {TOP_K}")
    print_plain(before)

    print(f"\nafter: rewrite, then top {CANDIDATE_K} reranked to {TOP_K}")
    print_reranked(after)


if __name__ == "__main__":
    main()

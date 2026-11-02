"""Load the Mealbox corpus, chunk it, embed it, and upsert it into Qdrant.

One chunk per entry, overlap 0. Point ids are a UUID5 of the entry id, so a
second run replaces the same points and the count stays at 48.

    docker compose up -d
    python ingest.py --fresh   # empty collection, then 48 points
    python ingest.py           # same 48 points, count unchanged

The category payload index and the top-3 similarity search are Part 3.
Vectors are embedded here because a Qdrant point cannot be stored without one.
"""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

CORPUS_PATH = Path("corpus/entries.json")
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "mealbox"

MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_SIZE = 384
UPSERT_BATCH = 64

# uuid5(NAMESPACE_URL, "mealbox:faq-001") is the same UUID on every machine.
# Qdrant point ids must be an integer or a UUID, so the human id stays in
# the payload as entry_id and this derived UUID is only the point id.
ID_PREFIX = "mealbox:"

REQUIRED_FIELDS = ("id", "kind", "title", "body", "category", "last_updated")
KINDS = {"faq", "ticket"}
CATEGORIES = {"account", "billing", "subscription", "delivery", "orders", "bugs"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest the Mealbox corpus into Qdrant.")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete the mealbox collection first, then load it from empty.",
    )
    return parser.parse_args()


def clean_text(text: str) -> str:
    """Trim the ends and collapse any run of whitespace to one space."""
    return " ".join(text.split())


def load_entries(path: Path = CORPUS_PATH) -> list[dict]:
    """Read corpus/entries.json and reject a row that would ingest badly."""
    if not path.exists():
        raise FileNotFoundError(f"Corpus not found at {path}. Run this from the repo root.")
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path} must be a non-empty JSON list")

    seen: set[str] = set()
    for entry in entries:
        missing = [field for field in REQUIRED_FIELDS if field not in entry]
        if missing:
            raise ValueError(f"Entry is missing {missing}: {entry.get('id', entry)}")
        entry_id = str(entry["id"]).strip()
        if not entry_id:
            raise ValueError("Entry id is empty")
        if entry_id in seen:
            raise ValueError(f"Duplicate entry id {entry_id}")
        seen.add(entry_id)
        if entry["kind"] not in KINDS:
            raise ValueError(f"{entry_id} has kind {entry['kind']!r}, expected faq or ticket")
        if entry["category"] not in CATEGORIES:
            raise ValueError(f"{entry_id} has unknown category {entry['category']!r}")
        datetime.strptime(entry["last_updated"], "%Y-%m-%d")
        if not clean_text(entry["title"]) or not clean_text(entry["body"]):
            raise ValueError(f"{entry_id} has an empty title or body after cleaning")
    return entries


def entry_to_chunk(entry: dict) -> dict:
    """Turn one corpus entry into one chunk.

    Each FAQ or ticket is one self-contained answer, about 450–1,050
    characters. A fixed-size splitter would cut the question in the title
    away from the rule in the body. Overlap only helps when a splitter
    cuts a long document mid-paragraph, and these entries are never cut.
    Chunk size is the whole entry. Overlap is 0.

    Revisit this if a later entry is a long thread that no longer fits
    in a single chunk.
    """
    title = clean_text(entry["title"])
    body = clean_text(entry["body"])
    return {
        "entry_id": str(entry["id"]).strip(),
        "title": title,
        "category": entry["category"],
        "last_updated": entry["last_updated"],
        "kind": entry["kind"],
        # Title first so a short question can match the line a customer would type.
        "text": f"{title}\n\n{body}",
    }


def chunk_entries(entries: list[dict]) -> list[dict]:
    return [entry_to_chunk(entry) for entry in entries]


def point_id_for(entry_id: str) -> str:
    """Stable Qdrant id. The same entry id always returns the same UUID."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{ID_PREFIX}{entry_id}"))


def embed_chunks(chunks: list[dict]) -> np.ndarray:
    print(f"Loading {MODEL_NAME} (first run downloads weights into ~/.cache)")
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    print(f"Embedding {len(chunks)} chunks")
    vectors = model.encode(
        [chunk["text"] for chunk in chunks],
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    if vectors.shape != (len(chunks), VECTOR_SIZE):
        raise ValueError(f"Expected vectors of shape {(len(chunks), VECTOR_SIZE)}, got {vectors.shape}")
    first_norm = float(np.linalg.norm(vectors[0]))
    print(f"vectors: shape={vectors.shape} dtype={vectors.dtype} first_norm={first_norm:.4f}")
    return vectors


def get_client() -> QdrantClient:
    client = QdrantClient(url=QDRANT_URL, timeout=10)
    try:
        client.get_collections()
    except Exception as exc:
        print(f"Cannot reach Qdrant at {QDRANT_URL}.")
        print("Start Docker Desktop, then from this folder run: docker compose up -d")
        print("Underlying error:", exc)
        raise SystemExit(1) from None
    return client


def point_count(client: QdrantClient) -> int:
    if not client.collection_exists(COLLECTION_NAME):
        return 0
    # exact=True avoids the approximate count Qdrant may return otherwise.
    return client.count(collection_name=COLLECTION_NAME, exact=True).count


def ensure_collection(client: QdrantClient) -> None:
    """Create the collection if it is missing. Never delete it here.

    Deleting on every run would also leave the count at 48, and would hide
    an upsert that was accidentally inserting new ids. --fresh is the only
    path that deletes.

    48 points sit under Qdrant's default full_scan_threshold (10,000), so
    search on this collection is exact. That is the same default you saw
    in the Qdrant capstone. The category payload index is added in Part 3;
    the payload stored here already has the category field.
    """
    if client.collection_exists(COLLECTION_NAME):
        print(f"Collection {COLLECTION_NAME} already exists; upserting in place")
        return
    print(f"Creating collection {COLLECTION_NAME} (cosine, {VECTOR_SIZE}-d)")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
    )


def upsert_chunks(client: QdrantClient, chunks: list[dict], vectors: np.ndarray) -> None:
    total = len(chunks)
    for start in range(0, total, UPSERT_BATCH):
        batch_chunks = chunks[start : start + UPSERT_BATCH]
        batch_vectors = vectors[start : start + UPSERT_BATCH]
        points = [
            models.PointStruct(
                id=point_id_for(chunk["entry_id"]),
                vector=vector.tolist(),
                payload={
                    "entry_id": chunk["entry_id"],
                    "title": chunk["title"],
                    "category": chunk["category"],
                    "last_updated": chunk["last_updated"],
                    "kind": chunk["kind"],
                    "text": chunk["text"],
                },
            )
            for chunk, vector in zip(batch_chunks, batch_vectors)
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"upserted {min(start + UPSERT_BATCH, total)}/{total}")


def main() -> None:
    args = parse_args()
    chunks = chunk_entries(load_entries())
    print(f"Loaded {len(chunks)} entries from {CORPUS_PATH}")
    print(f"{len(chunks)} chunks, one per entry, overlap 0")
    sample = chunks[0]
    print(
        f"sample payload: entry_id={sample['entry_id']} "
        f"point_id={point_id_for(sample['entry_id'])} "
        f"category={sample['category']} last_updated={sample['last_updated']}"
    )

    client = get_client()
    if args.fresh and client.collection_exists(COLLECTION_NAME):
        print(f"Deleting collection {COLLECTION_NAME}")
        client.delete_collection(COLLECTION_NAME)

    before = point_count(client)
    ensure_collection(client)
    vectors = embed_chunks(chunks)
    upsert_chunks(client, chunks, vectors)
    after = point_count(client)

    print(f"points before={before} after={after}")
    if after != len(chunks):
        print(f"Expected {len(chunks)} points, found {after}.")
        print("An older ingest may have left extra points. Re-run with --fresh.")
        raise SystemExit(1)
    if before == after:
        print("Point count unchanged. Upsert replaced the same point ids.")
    elif before == 0:
        print("Collection populated from an empty start.")


if __name__ == "__main__":
    main()

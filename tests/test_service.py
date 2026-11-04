"""Ingest idempotency, category filter, and abstention.

The abstain test patches the Groq token function. A live API key is not used.

    pytest
"""

from fastapi.testclient import TestClient

import main
from database import resources
from generate import ABSTAIN
from ingest import chunk_entries, load_entries, point_count, upsert_chunks
from retrieve import retrieve_ranked

WEDDING = (
    "Can Mealbox send a team to cook and serve dinner at my venue "
    "for a wedding of 80 guests?"
)
DIETITIAN = (
    "Can I book weekly calls with a Mealbox dietitian who will "
    "build a marathon-training menu for me?"
)
CAFE = "I run a cafe. Can I buy Mealbox ingredients wholesale and resell the cooked plates?"


def test_second_upsert_keeps_the_point_count():
    chunks = chunk_entries(load_entries())
    vectors = resources.embed([chunk["text"] for chunk in chunks])
    upsert_chunks(resources.qdrant, chunks, vectors)
    first = point_count(resources.qdrant)
    upsert_chunks(resources.qdrant, chunks, vectors)
    second = point_count(resources.qdrant)
    assert second == first
    assert second >= len(chunks)


def test_category_filter_drops_other_categories():
    _, _, _, ranked = retrieve_ranked(
        resources.qdrant,
        resources.embedder,
        "the chicken arrived warm",
        category="orders",
    )
    assert ranked
    categories = []
    entry_ids = []
    for _, _, point in ranked:
        payload = point.payload or {}
        categories.append(payload.get("category"))
        entry_ids.append(payload.get("entry_id"))
    assert categories == ["orders"] * len(categories)
    assert "ticket-006" in entry_ids


def test_unanswerable_questions_abstain_without_calling_groq(monkeypatch):
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("Groq was called")

    monkeypatch.setattr(main, "iter_answer_tokens", fail_if_called)
    monkeypatch.setattr(main, "groq_api_key", fail_if_called)
    client = TestClient(main.app)
    for question in (WEDDING, DIETITIAN, CAFE):
        response = client.post("/query", json={"question": question})
        assert response.status_code == 200
        body = response.text
        assert body.startswith('{"source_ids": []}')
        assert ABSTAIN in body

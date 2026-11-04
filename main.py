"""Mealbox support API.

POST /query streams an answer from the Part 4 retrieval path, or the
abstain sentence when the top cosine is below the Part 5 cutoff.
POST /ingest upserts one entry into the running collection.

    uvicorn main:app --reload
"""

from __future__ import annotations

import json
from typing import Annotated, Iterator

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from database import AppResources, get_resources
from generate import (
    ABSTAIN,
    MIN_COSINE,
    MissingGroqKey,
    format_context,
    groq_api_key,
    iter_answer_tokens,
)
from ingest import (
    entry_to_chunk,
    point_count,
    point_exists,
    point_id_for,
    upsert_chunks,
    validate_entry,
)
from retrieve import retrieve_ranked
from schemas import EntryCreate, IngestResponse, QueryRequest

app = FastAPI(title="Mealbox support RAG")

STREAM_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
STREAM_EXAMPLE = (
    '{"source_ids": ["faq-017"]}\n'
    "Pause from Plan. The length can be 1 to 8 weeks. [faq-017]"
)


def _validation_detail(exc: RequestValidationError) -> list[str]:
    lines = []
    for error in exc.errors():
        loc = [str(part) for part in error["loc"]]
        if loc and loc[0] == "body":
            loc = loc[1:]
        field = ".".join(loc) if loc else "request"
        message = str(error["msg"]).removeprefix("Value error, ")
        lines.append(f"{field}: {message}")
    return lines or ["Request is invalid."]


@app.exception_handler(StarletteHTTPException)
def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(RequestValidationError)
def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"detail": _validation_detail(exc)},
    )


@app.exception_handler(Exception)
def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "The service could not complete that request."},
    )


def _source_ids(ranked: list) -> list[str]:
    ids = []
    for _, _, point in ranked:
        entry_id = (point.payload or {}).get("entry_id")
        if entry_id:
            ids.append(str(entry_id))
    return ids


def _stream_abstain() -> Iterator[str]:
    yield json.dumps({"source_ids": []}) + "\n"
    yield ABSTAIN


def _stream_answer(question: str, ranked: list) -> Iterator[str]:
    yield json.dumps({"source_ids": _source_ids(ranked)}) + "\n"
    yield from iter_answer_tokens(question, format_context(ranked))


@app.post(
    "/query",
    responses={
        200: {
            "content": {"text/plain": {"example": STREAM_EXAMPLE}},
            "description": (
                "A stream. The first line is JSON with the entry ids sent to the model. "
                "The rest is the answer text. Below the cosine cutoff, source_ids is empty "
                "and the body is the abstain sentence."
            ),
        }
    },
)
def query(
    body: QueryRequest,
    resources: Annotated[AppResources, Depends(get_resources)],
):
    _, _, _, ranked = retrieve_ranked(resources.qdrant, resources.embedder, body.question)
    cosine = float(ranked[0][2].score) if ranked else None
    if cosine is None or cosine < MIN_COSINE:
        stream = _stream_abstain()
    else:
        try:
            groq_api_key()
        except MissingGroqKey as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from None
        stream = _stream_answer(body.question, ranked)
    return StreamingResponse(
        stream,
        media_type="text/plain; charset=utf-8",
        headers=STREAM_HEADERS,
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest_entry(
    entry: EntryCreate,
    resources: Annotated[AppResources, Depends(get_resources)],
):
    raw = entry.model_dump()
    try:
        entry_id = validate_entry(raw)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from None
    chunk = entry_to_chunk(raw)
    already_present = point_exists(resources.qdrant, entry_id)
    vectors = resources.embed([chunk["text"]])
    upsert_chunks(resources.qdrant, [chunk], vectors)
    return IngestResponse(
        entry_id=entry_id,
        point_id=point_id_for(entry_id),
        already_present=already_present,
        point_count=point_count(resources.qdrant),
    )

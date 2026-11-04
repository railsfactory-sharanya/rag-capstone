"""Answer a Mealbox question from reranked chunks, or abstain.

Uses the Part 4 retrieval path. If the top chunk's cosine is below
MIN_COSINE, the model is not called.

    python generate.py "How do I pause for a month?"
    python generate.py "Can I book weekly calls with a Mealbox dietitian who will build a marathon-training menu for me?"
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

from ingest import COLLECTION_NAME, MODEL_NAME, ensure_category_index, get_client
from retrieve import retrieve_ranked

# Highest unanswerable top cosine measured here was 0.4744 (dietitian question).
# Lowest answerable top cosine in that same check was 0.5309 (warm chicken).
MIN_COSINE = 0.50
ABSTAIN = "I don't have information on that."
GROQ_MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """You are Mealbox customer support.
Answer only using the context entries. Do not use outside knowledge.
Cite the entry id in square brackets for every claim, for example [faq-017].
If the entries do not state the answer, reply with exactly this sentence and nothing else:
I don't have information on that.
Do not turn a nearby article into a policy the entries never state."""


def format_context(rows: list) -> str:
    blocks = []
    for _, _, point in rows:
        payload = point.payload or {}
        blocks.append(
            f"[{payload.get('entry_id')}] category={payload.get('category')} "
            f"updated={payload.get('last_updated')}\n{payload.get('text')}"
        )
    return "\n\n".join(blocks)


class MissingGroqKey(RuntimeError):
    """Raised before any HTTP call when .env has no GROQ_API_KEY."""


def groq_api_key() -> str:
    load_dotenv(Path(__file__).resolve().parent / ".env")
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise MissingGroqKey("GROQ_API_KEY is missing. Put it in .env as GROQ_API_KEY=...")
    return api_key


def iter_answer_tokens(question: str, context: str):
    """Yield answer text as Groq produces it. Skips the separate reasoning field."""
    client = Groq(api_key=groq_api_key())
    stream = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0,
        stream=True,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nContext:\n{context}",
            },
        ],
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        text = getattr(delta, "content", None) if delta is not None else None
        if text:
            yield text


def generate_answer(question: str, context: str) -> str:
    try:
        return "".join(iter_answer_tokens(question, context)).strip()
    except MissingGroqKey as exc:
        raise SystemExit(str(exc)) from None


def main() -> None:
    parser = argparse.ArgumentParser(description="Answer from Mealbox chunks or abstain.")
    parser.add_argument("question", help="Customer question.")
    args = parser.parse_args()
    question = args.question.strip()
    if not question:
        raise SystemExit("Question is empty.")

    qdrant = get_client()
    if not qdrant.collection_exists(COLLECTION_NAME):
        raise SystemExit(f"Collection {COLLECTION_NAME} is missing. Run: python ingest.py --fresh")
    ensure_category_index(qdrant)

    print(f"Loading {MODEL_NAME}")
    embedder = SentenceTransformer(MODEL_NAME, device="cpu")
    rewritten, _, _, ranked = retrieve_ranked(qdrant, embedder, question)
    if not ranked:
        print(f"top: none threshold={MIN_COSINE:.2f}")
        print(f"answer: {ABSTAIN}")
        return

    _, _, top = ranked[0]
    payload = top.payload or {}
    cosine = float(top.score)
    print(
        f"top: entry_id={payload.get('entry_id')} cosine={cosine:.4f} "
        f"threshold={MIN_COSINE:.2f}"
    )
    if rewritten != question:
        print(f"rewrite: {rewritten}")
    if cosine < MIN_COSINE:
        print(f"answer: {ABSTAIN}")
        return

    print("answer:")
    print(generate_answer(question, format_context(ranked)))


if __name__ == "__main__":
    main()

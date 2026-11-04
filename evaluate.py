"""Score the Mealbox query path on eval/questions.json.

Uses prepare_query, so the source ids are the ones POST /query would stream.
An empty expected list means the corpus does not answer that question.

Context recall: share of expected ids present in those source ids.
Empty expected scores 1 only when the gate passes no ids.

Faithfulness: the answer cites at least one expected id, and every cited
id was in the source list. An empty expected list scores 1 only when the
answer is the abstain sentence and no ids were passed.

    python evaluate.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from database import resources
from generate import ABSTAIN, format_context, iter_answer_tokens
from main import prepare_query

QUESTIONS_PATH = Path(__file__).resolve().parent / "eval" / "questions.json"
CITATION = re.compile(r"\[\s*((?:faq|ticket)-\d{3})\s*\]")


def context_recall(expected: list[str], source_ids: list[str]) -> float:
    if not expected:
        return 1.0 if not source_ids else 0.0
    found = set(expected) & set(source_ids)
    return len(found) / len(set(expected))


def faithfulness(expected: list[str], source_ids: list[str], answer: str) -> float:
    if not expected:
        exact = answer.strip() == ABSTAIN and not source_ids
        return 1.0 if exact else 0.0
    cited = CITATION.findall(answer)
    if not cited:
        return 0.0
    if any(entry_id not in source_ids for entry_id in cited):
        return 0.0
    if not (set(cited) & set(expected)):
        return 0.0
    return 1.0


def answer_for(question: str, source_ids: list[str], ranked: list) -> str:
    if not source_ids:
        return ABSTAIN
    return "".join(iter_answer_tokens(question, format_context(ranked))).strip()


def top_label(ranked: list) -> str:
    if not ranked:
        return "none"
    rerank_total, _, point = ranked[0]
    entry_id = (point.payload or {}).get("entry_id")
    return (
        f"{entry_id} rerank={rerank_total:.4f} cosine={float(point.score):.4f}"
    )


def main() -> None:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    recalls = []
    faiths = []
    print(f"{len(questions)} questions from {QUESTIONS_PATH.name}")
    print()
    for item in questions:
        question = item["question"]
        expected = list(item["expected"])
        source_ids, ranked, _ = prepare_query(question, resources)
        answer = answer_for(question, source_ids, ranked)
        recall = context_recall(expected, source_ids)
        faith = faithfulness(expected, source_ids, answer)
        recalls.append(recall)
        faiths.append(faith)
        sources = ",".join(source_ids) if source_ids else "-"
        wanted = ",".join(expected) if expected else "-"
        print(
            f"{item['id']}  recall={recall:.0f}  faith={faith:.0f}  "
            f"top={top_label(ranked)}  sources={sources}  expected={wanted}"
        )
        print(f"    {question}")
        if recall < 1 or faith < 1:
            print(f"    answer: {answer}")
    print()
    print(f"context recall  {sum(recalls) / len(recalls):.3f}  ({sum(recalls):.0f}/{len(recalls)})")
    print(f"faithfulness    {sum(faiths) / len(faiths):.3f}  ({sum(faiths):.0f}/{len(faiths)})")


if __name__ == "__main__":
    main()

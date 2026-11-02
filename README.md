# Mealbox support RAG

A FastAPI service that answers Mealbox customer questions from a corpus of FAQs and resolved tickets. Mealbox is a fictional weekly meal-kit delivery. Retrieval will use Qdrant. This repository currently contains **Part 1: the corpus**.

## Corpus

Structured entries live in [`corpus/entries.json`](corpus/entries.json): 37 FAQs and 11 resolved tickets (48 total).

Each entry has an `id`, a `title`, a `body`, a `category`, and a `last_updated` date, plus a `kind` of `faq` or `ticket`.

[`corpus.md`](corpus.md) names three customer questions the corpus is supposed to leave unanswered, and lists the topic pairs that deliberately overlap.

Categories: `account`, `billing`, `subscription`, `delivery`, `orders`, `bugs`.

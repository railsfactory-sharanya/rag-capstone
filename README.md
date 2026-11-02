# Mealbox support RAG

A FastAPI service that answers Mealbox customer questions from a corpus of FAQs and resolved tickets. Mealbox is a fictional weekly meal-kit delivery. Retrieval uses Qdrant. Part 1 is the corpus. Part 2 loads, chunks, embeds, and upserts it.

## Corpus

Structured entries live in [`corpus/entries.json`](corpus/entries.json): 37 FAQs and 11 resolved tickets (48 total).

Each entry has an `id`, a `title`, a `body`, a `category`, and a `last_updated` date, plus a `kind` of `faq` or `ticket`.

[`corpus.md`](corpus.md) names three customer questions the corpus is supposed to leave unanswered, and lists the topic pairs that deliberately overlap.

Categories: `account`, `billing`, `subscription`, `delivery`, `orders`, `bugs`.

## Prerequisites

- Python 3.11+
- Docker Desktop running

The Qdrant capstone container also binds port 6333. Stop it before starting this one:

```bash
docker stop qdrant-capstone
```

## Run Qdrant

From this directory:

```bash
docker compose up -d
```

Dashboard: [http://localhost:6333/dashboard](http://localhost:6333/dashboard). Stop with `docker compose down`. The `./qdrant_storage` volume is gitignored.

## Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`.venv/` is gitignored.

## Part 2: Ingest

```bash
python ingest.py --fresh
python ingest.py
```

The first command creates the `mealbox` collection and upserts 48 points. The second command upserts the same point ids, so the count stays 48. Each entry is one chunk. The embedded text is the title plus the body, and the payload keeps `entry_id`, `title`, `category`, `last_updated`, and `kind`.

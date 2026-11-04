# Mealbox support RAG

A FastAPI service that answers Mealbox customer questions from a corpus of FAQs and resolved tickets. Mealbox is a fictional weekly meal-kit delivery. Retrieval uses Qdrant. Part 1 is the corpus. Part 2 loads, chunks, embeds, and upserts it. Part 3 adds a category payload index and a plain top-3 similarity search. Part 4 adds an optional category filter, a title-overlap rerank, and one query rewrite. Part 5 answers from those chunks, or refuses when the top score is too low. Part 6 serves that path from FastAPI.

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

The first command creates the `mealbox` collection and upserts 48 points. The second command upserts the same point ids, so the count stays 48. Each entry is one chunk. The embedded text is the title plus the body, and the payload keeps `entry_id`, `title`, `category`, `last_updated`, and `kind`. Ingest also creates a keyword payload index on `category`.

## Part 3: Raw similarity search

```bash
python search.py
python search.py "How do I pause for a month?"
```

Embeds that question with the same MiniLM model and prints the top 3 points with score and payload. No category filter and no reranker. The default query should bring back `faq-017`.

## Part 4: Retrieval

```bash
python retrieve.py
python retrieve.py --category orders "the chicken arrived warm"
```

The default question is one plain search gets wrong. The script prints that top 3, then the top 3 after a query rewrite and a title-overlap rerank. `--category` limits the second search to one category. The measured before/after is in [`retrieval-notes.md`](retrieval-notes.md).

## Part 5: Grounded answers

Put a Groq key in `.env` as `GROQ_API_KEY=...`. That file is gitignored.

```bash
python generate.py "How do I pause for a month?"
python generate.py "Can Mealbox send a team to cook and serve dinner at my venue for a wedding of 80 guests?"
```

The script retrieves with the Part 4 path. If the top chunk's cosine is below 0.50, it prints `I don't have information on that.` and does not call the model. Otherwise Groq (`openai/gpt-oss-120b`) writes an answer that cites entry ids.

## Part 6: FastAPI

Start Qdrant first (`docker compose up -d`). The `mealbox` collection from Part 2 has to be loaded. From this directory, with the venv active:

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

The first start loads MiniLM once. `--reload` loads it again each time a file changes. For a review demo, `uvicorn main:app` keeps that one copy for the life of the process.

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). Both `POST /query` and `POST /ingest` are listed.

Stream a question (`-N` shows tokens as they arrive):

```bash
curl -N -X POST http://127.0.0.1:8000/query \
  -H 'content-type: application/json' \
  -d '{"question":"How do I pause for a month?"}'
```

The first line is JSON, `{"source_ids": ["faq-017", ...]}`, the entry ids of the chunks sent to the model. The rest of the body is the answer. The same call with one of the three questions in `corpus.md` returns `{"source_ids": []}` and then `I don't have information on that.`

An empty question is a 422:

```bash
curl -s -X POST http://127.0.0.1:8000/query \
  -H 'content-type: application/json' \
  -d '{"question":""}'
```

Add one entry without restarting. Use a new id. Posting an existing id overwrites that point.

```bash
curl -s -X POST http://127.0.0.1:8000/ingest \
  -H 'content-type: application/json' \
  -d '{"id":"faq-099","kind":"faq","title":"Test entry","body":"This is only a live-demo row.","category":"account","last_updated":"2026-11-03"}'
```

This writes Qdrant only. `corpus/entries.json` is unchanged. `python ingest.py` puts the original 48 texts back and leaves `faq-099` in place. `python ingest.py --fresh` reloads the file and drops `faq-099`.

`python search.py`, `python retrieve.py`, and `python generate.py` still run on their own. They do not go through this server.

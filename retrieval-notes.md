# Retrieval notes

Part 4 before/after for one query that plain similarity search got wrong.

Collection: `mealbox`. Model: `all-MiniLM-L6-v2` on CPU, same model as ingest. Score is cosine similarity.

## The query

```text
I am traveling for a month and do not want boxes
```

The right article is `faq-017`, "How do I pause Mealbox for several weeks while I am away?" A month fits the pause rule of 1 to 8 weeks. `faq-018` is one week. `faq-016` ends the plan.

## Before: plain top 3

`search.py` and the "before" block of `retrieve.py` embed that sentence and take the nearest 3 points. No filter, no rewrite, no rerank.

| Rank | Score | Entry | Why it came back |
|------|-------|-------|------------------|
| 1 | 0.4791 | `faq-029` | Title is about PO boxes and where Mealbox delivers. The word "boxes" pulls a shipping article. |
| 2 | 0.4022 | `faq-028` | Nobody is home when the box arrives. |
| 3 | 0.3909 | `faq-036` | Switch the delivery day between Tuesday and Thursday. |

`faq-017` is rank 6 at 0.3750, just under `faq-022` ("what comes inside a Mealbox"). Plain top 3 never shows the pause article.

## Category filter does not fix this query

Restricting the same vector to `category=subscription` removes the delivery articles. The new top is `faq-022` at 0.3841, then `faq-017` at 0.3750. The nearest subscription article is still "what is in the box," because the query never says "pause." The payload index is doing its job. The missing piece is the wording.

## After: query rewrite, then rerank

`rewrite_query` appends a clause when the question is about travel and a month or weeks:

```text
I am traveling for a month and do not want boxes. pause the weekly plan for several weeks while away
```

That string is embedded with the same model. Qdrant returns 10 candidates. `rerank` adds `0.15 * title_overlap` to the cosine score. Overlap is the share of query tokens, stopwords removed, that also appear in the title.

| Rank | Rerank | Cosine | Title overlap | Entry |
|------|--------|--------|---------------|-------|
| 1 | 0.7890 | 0.7140 | 0.50 | `faq-017` pause |
| 2 | 0.6268 | 0.6161 | 0.07 | `faq-018` skip one week |
| 3 | 0.6267 | 0.6052 | 0.14 | `faq-016` cancel |

`faq-017` is first. Skip and cancel follow it, which is the right neighborhood: those are the other ways to stop a week, and they are no longer hidden behind PO-box articles.

Rerank does not create this fix. On the original sentence, `faq-017` is outside the top 3, and the word "boxes" overlaps the wrong titles, so a title rerank would keep `faq-029` on top. The rewrite puts `faq-017` into the candidate list. The title overlap then agrees with that order, because "pause," "several," "weeks," and "away" are in its title.

A keyword hybrid would have made the miss worse. "boxes" is the token plain search already over-weighted.

## What the rerank changes on its own

On "I was charged twice for the same week" the rewrite does not fire. Plain order is `faq-015` (0.5244), then `ticket-002` (0.4738). After rerank, `ticket-002` is first because every content word of the question is in that ticket's title (`title_overlap=1.00`, rerank 0.6238). `faq-015` drops to second (overlap 0.20, rerank 0.5544). Both articles answer the question. The heuristic prefers the title that repeats the question.

## Category filter, separate check

`--category` is applied only to the candidate search in the "after" block. The "before" block stays unfiltered so the baseline does not move.

```bash
python retrieve.py --category orders "the chicken arrived warm"
```

Before: `ticket-006` (orders), `faq-026` (delivery), `faq-022` (subscription).

After: `ticket-006`, then other `orders` points only. `faq-026` and `faq-022` are gone. The keyword index on `category` is what makes that filter a lookup rather than a scan of every payload.

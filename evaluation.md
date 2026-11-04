# Evaluation

Two runs of `python evaluate.py` on the same 15 questions in `eval/questions.json`. Both use `prepare_query`, the same gate as `POST /query` and `generate.py`.

Collection: `mealbox`. Model: `all-MiniLM-L6-v2` on CPU.

## The change

The cutoff used to read the cosine on the reranked winner and ignore the rerank total. It now compares 0.50 to that total, `cosine + 0.15 × title overlap`, which is the number `rerank` already sorts on. `prepare_query` and `generate.py` both use it. `search.py` and `retrieve.py` are unchanged. They print neighbors and do not abstain.

## Scores

| Metric | Before | After |
|--------|--------|-------|
| Context recall | 0.800 (12/15) | 0.933 (14/15) |
| Faithfulness | 0.800 (12/15) | 0.933 (14/15) |

Context recall is the share of expected entry ids present in the source list the API would stream. An empty expected list scores 1 only when that list is empty.

Faithfulness scores 1 when the answer cites at least one expected id and every `[faq-###]` or `[ticket-###]` citation was in that source list. Spaces inside the brackets count. An empty expected list scores 1 only when the answer is exactly `I don't have information on that.`

Both metrics moved together. The misses were questions where the right chunk was rank 1 and then discarded before Groq ran, so the answer could not cite it. A prompt edit would not have raised either score.

## What changed row by row

| Id | Question | Before | After |
|----|----------|--------|-------|
| q11 | My box arrived in the middle of the night | Abstain. `ticket-004` cosine 0.4368. | Pass. Rerank total 0.5618. Cites `ticket-004`. |
| q12 | I was charged twice for the same week | Abstain. `ticket-002` cosine 0.4738. | Pass. Rerank total 0.6238. Cites `ticket-002`. |
| q05 | I will be away for several weeks and do not want a box | Abstain. Top is `faq-029` at cosine 0.4487. | Still abstain. Rerank total 0.4654. The winner is still the PO-box article. |

The wedding, dietitian, and cafe questions stay refused. Their rerank totals are 0.4227, 0.4851, and 0.4425. The dietitian question is still the closest.

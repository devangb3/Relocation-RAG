# Retrieval evaluation

## Questions to try against the index

1. **My approved move was completed 45 days ago. Under the current policy, am I still within the filing window?** No. REL-103.3 in the 2026 edition gives 30 calendar days. The 2025 edition gives 60.
2. **For HA-217, how much of the packing receipt can I claim after the shuttle amendment?** $410. Amendment A1 released $4,650, and the carrier's posted direct bill was $4,240. The employee's $460 purchase exceeds the balance. See REL-201.2, REL-201.3, REL-104.2 and REL-102.1.
3. **What is due to the employee on HA-271 after the advance?** $1,240. The eligible lease release is $1,300 after the deposit credit. Two capped hotel nights contribute $300 and the flight and bag contribute $400. Subtract the $760 advance once. See REL-202.1–3, REL-106.2, REL-102.3–4 and REL-105.3.
4. **Was WI-084 filed on time even though the replacement folio arrived in March?** Yes. The corrected December 29, 2025 completion date selects the 2025 edition. February 20 is within the window ending February 27. The March attachment corrects the same timely filed line. See REL-203.1–2 and the 2025 REL-103.3–4.
5. **For BE-119, how much of the furnished apartment week is eligible?** $840. Divide $980 by seven invoiced nights, then count six occupied nights at $140. The deposit and housekeeping are separate. See REL-204.1–2, REL-107.2 and REL-102.4.
6. **How much did JU-510 overpay after the hotel gave back $45?** $11. The first night originally paid $150 under the shared nightly cap. Its revised eligible cost is $139. The other night is unaffected. See REL-206.1–2, REL-107.1 and REL-103.13.
7. **Did Treasury receive the money back on ME-046?** The record supports a $220 return due ($300 less $80), but does not establish receipt. See REL-207.1–2 and REL-105.3.
8. **Which policy mentions ADV-4?** REL-105.1, “Requesting cash before departure,” identifies ADV-4 as the form for requesting a relocation advance. On the same index with a ten-result limit, hybrid retrieval returned REL-105.1 second. Vector-only retrieval missed it in the top ten.
9. **Who can sign a single-source note?** Purchasing may sign one when only one carrier serves the route. The note records which carriers were contacted. See REL-104.1. With the same vector index and a ten-result limit, BM25 hybrid retrieval ranked the passage fourth. The earlier substring-based hybrid missed it in the top ten. Its keyword search looked for the whole question or a policy code. BM25 could match the individual terms in “single-source note.”

## Retrieval baseline before reranking

On 2026-11-19, the default BM25 + MiniLM hybrid retriever at commit `201a0fb` scored **76.06% macro Recall@10** at the ten-chunk generation limit across the nine questions above. Each query ran once with a candidate limit of 10 against the local Chroma index. All 140 indexed chunks matched the five policy files. Each cutoff uses the first *k* chunks of that same ranked list.

| Question | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
| --- | ---: | ---: | ---: | ---: |
| 1. Current filing window | 0/1 | 1/1 | 1/1 | 1/1 |
| 2. HA-217 packing receipt | 1/4 | 2/4 | 2/4 | 2/4 |
| 3. HA-271 amount due | 1/7 | 3/7 | 3/7 | 3/7 |
| 4. WI-084 filing date | 1/4 | 2/4 | 2/4 | 2/4 |
| 5. BE-119 apartment | 1/4 | 2/4 | 3/4 | 4/4 |
| 6. JU-510 hotel refund | 1/4 | 2/4 | 3/4 | 3/4 |
| 7. ME-046 Treasury receipt | 1/3 | 1/3 | 1/3 | 2/3 |
| 8. ADV-4 form | 0/1 | 1/1 | 1/1 | 1/1 |
| 9. Single-source note | 0/1 | 0/1 | 1/1 | 1/1 |
| **Macro average** | **16.40%** | **52.91%** | **69.58%** | **76.06%** |

Each fraction counts distinct cited evidence sections represented in the first *k* chunks, divided by the number required for that question. Multiple chunks from one section earn one hit. The macro average gives each question equal weight. Q1 requires the current (2026) filing window. Q4 requires the 2025 claims sections. Q6 accepts either edition of REL-103.13 because both give the same refund rule. A section hit measures retrieval coverage only and can come from a chunk that contains just part of the needed evidence.

Reordering the same ten candidates leaves Recall@10 fixed. For example, HA-271 has only three of its seven required sections anywhere in the ten candidates.

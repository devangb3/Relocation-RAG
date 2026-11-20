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

## Local cross-encoder results

All nine questions were run against the 140-chunk local index. The cross-encoder scored each of 30 hybrid candidates using the question, source filename, effective date, and passage. The final ten chunks are the generation context. The two 30-candidate runs use the same hybrid search configuration.

| Ranking | Macro Recall@1 | Recall@3 | Recall@5 | Recall@10 | Questions with all sections at 10 | Mean search time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hybrid, 10 candidates (rerun) | 16.40% | 50.13% | 69.58% | 76.06% | 4/9 | 6 ms |
| Hybrid, first 10 of 30 | 14.81% | 46.96% | 54.10% | 70.50% | 3/9 | 6 ms |
| Cross-encoder, top 10 of 30 | 49.74% | 67.72% | 73.28% | **81.35%** | **5/9** | 128 ms |

At the ten-chunk generation limit, reranking gained **5.29 percentage points** over the original baseline. HA-271 recovered four of seven sections (previously three), and ME-046 recovered all three (previously two). The other seven questions retained their baseline section counts. The hybrid rerun reproduced the historical Recall@10, with a different Recall@3. Timings are local means over nine sequential queries after one warm-up per configuration. They exclude model loading and answer generation.

## Live answer quality after reranking

 `generate_answer` function tested each of the nine questions. Seven answers gave the expected conclusion, and two abstained on a requested amount. Four answers explicitly named a supporting REL section (Q4, Q7, Q8 and Q9). This run evaluated the reranked pipeline only.

| Question | Review against the answer key |
| --- | --- |
| Q1. Filing window | Correctly distinguished the current 30-day window from the previous 60-day window. |
| Q2. HA-217 | Correctly subtracted the posted $4,240 bill from the released $4,650 budget. |
| Q3. HA-271 | Identified the $760 advance. The retrieved context lacked the lease, travel and lodging rules needed to calculate the $1,240 payable. |
| Q4. WI-084 | Cited the February 20 confirmation and March correction on the same line. Omitted the 2025 filing window and February 27 deadline. |
| Q5. BE-119 | Correctly applied $140 to six occupied nights and excluded the deposit and housekeeping. |
| Q6. JU-510 | Calculated the revised $139 charge. The context included an example using $150 but lacked the general household cap and split-room rule needed to establish the $11 overpayment. |
| Q7. ME-046 | Correctly distinguished the $220 owed from evidence of repayment and cited the register. |
| Q8. ADV-4 | Identified the correct policy and section title. |
| Q9. Single-source note | Identified the signer, the single-carrier condition and the section supporting both. |

## Reranking vs hybrid-only retrieval

On 2026-11-19, the same nine questions went through the same generation function, prompt and Foundry deployment with the original hybrid top ten from the 140-chunk index. A conclusion counted as correct when the answer gave the requested determination or eligible amount. On Q1, listing both policy windows and declining to choose earned partial credit only. On Q5, giving the requested eligible amount counted even with a caveat about final payment.

| Question | Hybrid only | With reranking |
| --- | --- | --- |
| Q1. Current filing window | Gave both windows and declined to choose an edition | Answered no under the current 30-day window |
| Q2. HA-217 | $410 | $410 |
| Q3. HA-271 | Abstained, missing the lease settlement and eligibility rules | Abstained, retrieved the settlement but still lacked eligibility rules |
| Q4. WI-084 | Yes, based on the timely line and replacement attachment | Same conclusion, cited REL-203.2. Both omitted the February 27 deadline |
| Q5. BE-119 | $840 eligible, with a caveat about the lodging budget | $840 eligible and confirmed it fell within the $1,050 budget |
| Q6. JU-510 | Reached $139 for the revised charge, abstained on overpayment | Same outcome. Both missed the $11 overpayment |
| Q7. ME-046 | Correctly said receipt was unconfirmed | Also calculated the $220 due and cited the register |
| Q8. ADV-4 | REL-105.1 | REL-105.1 |
| Q9. Single-source note | Purchasing, with the correct condition | Same signer and condition |

Hybrid-only gave six direct correct conclusions, one conditional answer and two abstentions. Reranking gave seven direct correct conclusions and two abstentions. Three hybrid-only answers named a supporting REL section (Q1, Q8, Q9), compared with four reranked answers (Q4, Q7, Q8, Q9). With one generation per configuration.

## Diagnosing the conflicting filing deadlines

**Question asked on November 19, 2026**  
“My approved move was completed 45 days ago. Under the current policy, am I still within the filing window?”

**Earlier answer from the full pipeline**

> **If your recorded move completion date was in 2026, no**: the filing window is 30 calendar days, so at 45 days it has passed. If your recorded completion date was in 2025, you are still within that edition’s 60-day window. The edition in force on the completion date governs, even if you file in the following year. [REL-103.3, 2026 and 2025 editions]
>
> Your exact recorded completion date is needed to determine which rule applies. People Operations records that date as when you arrived at the destination to take up residence. [REL-101.4]

**Expected answer.** No. Under the current edition, day 45 is 15 days past the deadline. 

**1. Did retrieval find the governing passage?** Yes, both passages reached the initial generation context with their filenames and effective dates.

**2. Where did the competing answer come from?** The same section says “within 30 calendar days” in the edition effective January 1, 2026 and “within 60 calendar days” in the edition effective January 1, 2025. Both select the edition by the recorded move completion date. Day 45 falls between those limits, so an outdated copy can reverse the answer to a current-policy question. [2026 source, REL-103.3](../data/policies/rel-103-claims-2026.md#rel-1033-filing-window) · [2025 source, REL-103.3](../data/policies/rel-103-claims-2025.md#rel-1033-filing-window)

**Diagnostic check.** We removed the 2025 claims chunks from a temporary copy of the index and repeated the question. The 2026 rule still ranked first, and “60 calendar days” disappeared from the top ten. This traced the competing deadline to the older handbook.

**Remediation.** We kept both editions in the index and recorded the dates when each governs a claim. Before ranking passages, vector search and BM25 now filter policy chunks to the editions in force on the lookup date. The initial lookup uses today’s date, so a current-policy question gets the 30-day rule. The initial search can retrieve the transfer record for a historical claim. For WI-084, the model can read the corrected completion date from the register and pass `as_of="2025-12-29"` to `search_corpus` to retrieve the 60-day rule. Follow-up searches inherit that date.

**Verification.** On the rebuilt index, vector, hybrid, and reranked searches each returned only the 2025 claims edition for December 31, 2025, and only the 2026 claims edition for January 1, 2026. We then verified both answers.

- **Current policy, day 45.** The assistant answered “No” and cited REL-103.3 for the 30-calendar-day limit. It gave the employee a direct answer without introducing the historical deadline.
- **WI-084.** The initial search returned the register entries REL-203.2 and REL-203.1. The model then searched with `as_of="2025-12-29"`, and the 2025 edition of REL-103.3 ranked first. It concluded that the February 20 filing met the February 27 deadline and explained that the March 4 replacement folio retained the original filing date.

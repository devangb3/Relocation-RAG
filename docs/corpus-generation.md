# Part 1 corpus generation

## Generation prompt

Write a fixed Markdown corpus for North Alder Instruments, a fictional employer. Address employees making company-requested transfers between US offices. Produce four English policy documents of 500–800 words each. Use dated editions and numbered, descriptive sections that an assistant can cite. Give employees practical examples and use the same names for roles, records, and dates throughout.

Write three current policies effective January 1, 2026. REL-101 governs eligibility and written preapproval. REL-102 governs covered moving expenses, travel, lodging, limits, and exclusions. REL-103 governs receipts, submission, and review. Define the move completion date consistently. Put the filing window only in REL-103. Use 30 calendar days after completion in the current claims policy. Make the fourth document the January 1, 2025 edition of REL-103, with the same provisions except for a filing window of 60 calendar days. Include the same section headings in both editions. Avoid calling attention to the conflict in the policies. Keep these documents as fixed inputs to retrieval.

## Planted issue and expected evidence

Use only the four files under `data/policies/` as source documents. An older claims policy was retained alongside the current one. Both editions contain a section titled “REL-103.3 Filing window.” The 2025 edition allows 60 calendar days after move completion. The 2026 edition allows 30. All other claims provisions are identical.

Diagnostic question for the later evaluation: “My approved move was completed 45 days ago. Under the current policy, am I still within the filing window?” The expected answer is no, citing REL-103.3 in the 2026 edition. A yes citing the 2025 edition would expose the stale source. This is an expected failure scenario; record the actual retrieved passages and answer when running the pipeline.

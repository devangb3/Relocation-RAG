# Relocation RAG

An internal Q&A assistant over a company relocation reimbursement policy.

The policies are chunked, embedded locally, and stored in Chroma. The assistant retrieves relevant passages by vector similarity and uses Microsoft Foundry to answer with policy citations.

Dependencies are managed with [uv](https://docs.astral.sh/uv/). Continuous integration installs the locked environment and runs the test suite.

## Setup

Requires Python 3.13 or newer and uv.

```sh
uv sync --locked
```

## Index the policies

From the repository root, run `uv run --locked python -m relocation_rag.index`.
The chunker groups paragraphs within each numbered section, targeting 128 words
and keeping individual paragraphs and Markdown tables intact. Each chunk includes
its section heading. Chroma stores the text, embedding, filename, section ID,
and effective date in the local `.chroma` directory under `policy_chunks`.

## Ask a policy question

Set `FOUNDRY_BASE_URL`, `FOUNDRY_API_KEY`, and `FOUNDRY_MODEL` in `.env`, using
[.env.example](.env.example) as a guide. Use your Foundry resource URL, with or
without `/openai/v1`, and the name of a model deployment that supports the
Responses API. [Microsoft Foundry Responses API](https://learn.microsoft.com/azure/foundry/openai/how-to/responses)

```sh
uv run --locked --env-file .env python -m relocation_rag.generate_answer \
  "My approved move was completed 45 days ago. Under the current policy, am I still within the filing window?"
```

The CLI prints the answer to stdout and logs token usage to stderr. Set both
`FOUNDRY_INPUT_USD_PER_MILLION` and `FOUNDRY_OUTPUT_USD_PER_MILLION` to log an
estimated cost using your deployment's rates.

## Tests

```sh
uv run pytest
uv run ruff check .
```

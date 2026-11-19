# Relocation RAG

An internal Q&A assistant over a company relocation reimbursement policy.

The policies are chunked, embedded locally, and stored in Chroma. The assistant combines vector similarity with keyword search and uses Microsoft Foundry to answer with policy citations.

Dependencies are managed with [uv](https://docs.astral.sh/uv/). Continuous integration installs the locked environment and runs the test suite.

## Run with Docker Compose

From the repository root, build the image, index the bundled policies, and run a
ranked search. These commands require Docker with Compose and internet access
for the initial dependency and model downloads.

```sh
docker compose build
docker compose run --rm rag relocation_rag.index
docker compose run --rm rag relocation_rag.retrieval "Which policy mentions ADV-4?"
```

The application runs as a non-root CLI process. Each command exits when it
finishes. Compose keeps the Chroma index and Hugging Face model cache in named
volumes shared across runs. After editing the bundled policies, rebuild the
image and rerun indexing. The indexer updates files present in the corpus.
If you remove or rename a policy, rebuild the index in a fresh Chroma volume
to discard records for the old filename.

To generate an answer, copy `.env.example` to `.env`, fill in your Foundry
deployment details, and pass that file to the answer command at runtime:

```sh
docker compose run --rm --env-from-file .env rag relocation_rag.generate_answer \
  "Which form do I need for a relocation advance?"
```

## Local Python setup

Requires Python 3.13 or newer and uv.

```sh
uv sync --locked
```

## Index the policies

From the repository root, run `uv run --locked python -m relocation_rag.index`.
`sentence-transformers/all-MiniLM-L6-v2` from Hugging Face is used as the local embedding model for both passages and queries.

The chunker targets 160 MiniLM tokens per chunk, including the section heading and
special tokens. It carries over the previous prose sentence within the same
section when the sentence is at most 32 tokens and fits in the next chunk.
Every table row becomes one sentence pairing its column names with its values.
Consecutive row sentences share a chunk when they fit the target. Each chunk
starts with the section heading. 

Run the indexing command before querying. The text, embedding,
filename, section ID, and effective date are stored in the
`policy_chunks` collection under `.chroma`.

## Hybrid retrieval

Hybrid search computes BM25 scores over the current Chroma documents in memory
for each query. It matches individual words case-insensitively and treats codes
such as ADV-4 and REL-105.1 as whole tokens.
Chunks found by both searches come first, in vector order. Keyword-only chunks
follow in BM25 score order (ties use chunk ID), then the remaining vector hits in their original order.
Each chunk appears once.

Answer generation retrieves up to 30 hybrid candidates and passes the ten highest
scoring chunks to Foundry. A local cross-encoder
(`cross-encoder/ms-marco-MiniLM-L6-v2`) scores the question against each passage,
including the source filename and effective date. Ties retain the hybrid order.
Foundry can answer immediately or call `search_corpus` with a question derived
from the passages. Each call runs the same local retrieval pipeline and returns
cited passages. After three follow-up searches, the model must answer with the
available evidence and identify any unresolved gap.

Inspect the same ranking with `uv run --locked python -m relocation_rag.retrieval "Which policy mentions ADV-4?"`.
Use `--hybrid-only` or `--vector-only` for the earlier rankings. The CLI's
`--limit` controls the number returned (default 10). In reranked mode the default k = 30.

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

Run pytest coverage

```sh
uv run --with pytest-cov pytest --cov=relocation_rag --cov-report=term-missing
```

To run the nine-question policy evaluation, build the index and configure Foundry
as described above, then run:

```sh
uv run --locked --env-file .env pytest -q -s evaluation
```

Each question makes one to four live requests for answer generation and one separate
judging request. The Foundry deployment must support function calling. The judge requires structured JSON output support and returns a yes/no verdict and reason
for each rubric item. A case passes when every verdict is yes. To measure only
local retrieval, run `uv run --locked pytest -q -s evaluation -k recall`.
Each recall test prints the fraction of required evidence groups found in the initial top
ten and fails only when that count is zero. Each answer test checks the generated text for
the case's required facts. The live suite runs separately from the default unit
tests. See the [recorded evaluation results](docs/retrieval-evaluation.md#automated-pytest-evaluation).

# Relocation RAG

An internal Q&A assistant over a company relocation reimbursement policy.

The policy is chunked, embedded locally, and stored for retrieval. A question is answered from the most relevant passages, using vector search together with keyword search, then reranked. Each answer cites the policy section it came from. A fixed evaluation set checks that the right passages are retrieved and that answers include the expected policy details.

Dependencies are managed with [uv](https://docs.astral.sh/uv/). Continuous integration installs the locked environment and runs the test suite.

## Setup

Requires Python 3.13 or newer and uv.

```sh
uv sync --locked
```

## Tests

```sh
uv run pytest
uv run ruff check .
```

FROM ghcr.io/astral-sh/uv:0.12.24 AS uv
FROM python:3.13-slim-bookworm

WORKDIR /app
COPY --from=uv /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project --no-cache --python /usr/local/bin/python

ENV PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/home/rag/.cache/huggingface \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
RUN useradd --create-home --uid 10001 rag \
    && install -d -o rag -g rag /app/.chroma "$HF_HOME"
COPY core/ core/
COPY data/policies/ data/policies/
USER rag
ENTRYPOINT ["python", "-m"]
CMD ["core.retrieval", "--help"]

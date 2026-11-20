import logging
from dataclasses import asdict
from pathlib import Path

from relocation_rag.chunking import chunk_policy
from relocation_rag.embeddings import get_embedding_model, get_policy_collection
from relocation_rag.logging_service import LoggingService

logger = logging.getLogger("relocation_rag.index")

def index_policy(path: Path) -> None:
    model = get_embedding_model()
    chunks = chunk_policy(path, model.tokenizer, max_tokens=model.max_seq_length)
    ids = [chunk.id for chunk in chunks]
    texts = [chunk.text for chunk in chunks]
    vectors = model.encode(texts, normalize_embeddings=True)
    collection = get_policy_collection()
    previous_ids = collection.get(where={"source": path.name}, include=[])["ids"]
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=vectors,
        metadatas=[
            {k: v for k, v in asdict(c).items() if k not in ("id", "text") and v is not None}
            for c in chunks
        ],
    )
    stale_ids = set(previous_ids).difference(ids)
    if stale_ids:
        collection.delete(ids=sorted(stale_ids))
    logger.info("Indexed %s chunks from %s.", len(chunks), path.name)


def main() -> None:
    LoggingService()
    for path in Path("data/policies").glob("*.md"):
        index_policy(path)
    logger.info("Indexed all policies.")


if __name__ == "__main__":
    main()

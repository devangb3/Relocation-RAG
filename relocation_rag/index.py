from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from relocation_rag.chunking import chunk_policy

#Q1: How is indexing done? Here and what are some other ways to index?
def index_policy(path: Path) -> None:
    chunks = chunk_policy(path)
    texts = [chunk.text for chunk in chunks]
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vectors = model.encode(texts) #Vectors of size 384
    collection = chromadb.PersistentClient(path=".chroma").get_or_create_collection("policy_chunks")
    collection.upsert(
        ids=[chunk.id for chunk in chunks],
        documents=texts,
        embeddings=vectors,
        metadatas=[
            {"source": c.source, "section": c.section, "effective_date": c.effective_date}
            for c in chunks
        ],
    )
    print(f"Indexed {len(chunks)} chunks from {path.name}.")

def main() -> None:
    for path in Path("data/policies").glob("*.md"):
        index_policy(path)
    print("Indexed all policies.")


if __name__ == "__main__":
    main()

"""Prove the embed/store/query loop on two unrelated records."""
import os
import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
collection = chromadb.EphemeralClient().create_collection("two_records")
texts = ["Book a desk through the seating calendar.",
         "Submit a receipt to reclaim travel expenses."]
for key, passage in zip(["desk", "travel"], texts):
    collection.add(ids=[key], documents=[passage],
                   embeddings=model.encode([passage], normalize_embeddings=True))
    print(f"Stored {key} | collection size = {collection.count()}")

question = "How do I get paid back for a business trip?"
hits = collection.query(query_embeddings=model.encode([question], normalize_embeddings=True),
                        n_results=2)
print("Question:", question)
for place, (key, doc) in enumerate(zip(hits["ids"][0], hits["documents"][0]), 1):
    print(f"{place}. {key}: {doc}")
assert hits["ids"][0][0] == "travel"

import chromadb
from sentence_transformers import SentenceTransformer

def retrieve_policy(query: str) -> list[str]:
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    query_vector = model.encode(query) #Vectors of size 384
    collection = chromadb.PersistentClient(path=".chroma").get_or_create_collection("policy_chunks")
    results = collection.query(
        query_embeddings=query_vector,
        n_results=10, #Number of results to return
        include=["distances", "documents", "metadatas"], #Include distances, documents, and metadatas
    )
    return results

def main() -> None:
    query = "What is the policy on relocation?"
    results = retrieve_policy(query)
    print(results)

if __name__ == "__main__":
    main()
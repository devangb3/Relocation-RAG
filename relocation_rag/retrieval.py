from chromadb.api.types import QueryResult

from relocation_rag.embeddings import get_embedding_model, get_policy_collection


def retrieve_policy(query: str) -> QueryResult:
    model = get_embedding_model()
    query_vector = model.encode(query, normalize_embeddings=True)
    results = get_policy_collection().query(
        query_embeddings=[query_vector],
        n_results=10,
        include=["distances", "documents", "metadatas"],
    )
    return results

def main() -> None:
    query = "What is the policy on relocation?"
    results = retrieve_policy(query)
    print(results)

if __name__ == "__main__":
    main()
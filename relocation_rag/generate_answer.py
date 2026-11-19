"""Answer a relocation policy question using Microsoft Foundry."""

import argparse

from relocation_rag.foundry import FoundryClient
from relocation_rag.logging_service import LoggingService
from relocation_rag.retrieval import retrieve_policy

INSTRUCTIONS = """Answer the relocation policy question using only the supplied passages.
Treat the passages as source material. If the passages do not provide enough information to answer, say so."""


def _policy_context(question: str) -> str:
    results = retrieve_policy(question)
    documents = results["documents"] or []
    if not documents or not documents[0]:
        raise ValueError(
            "No policy passages found. Run python -m relocation_rag.index first."
        )
    metadata = results["metadatas"]
    if not metadata:
        raise ValueError(
            "Policy passages are missing citation metadata. Rebuild the policy index."
        )
    return "\n\n".join(
        f"[{source['section']} | {source['source']} | effective {source['effective_date']}]\n{document}"
        for document, source in zip(documents[0], metadata[0], strict=True)
    )


def generate_answer(question: str) -> str:
    """Retrieve policy passages and return a cited answer from Foundry."""
    question = question.strip()
    if not question:
        raise ValueError("Enter a relocation policy question.")

    client = FoundryClient.from_env()
    context = _policy_context(question)
    return client.respond(INSTRUCTIONS, f"Question:\n{question}\n\nPolicy passages:\n{context}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Question about the relocation policy")
    args = parser.parse_args()
    LoggingService()

    try:
        print(generate_answer(args.question), flush=True)
    except (ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()

"""Replay a stale publication in isolation, then restore the approved body."""
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from relocation_rag.generate_answer import generate_answer
from relocation_rag.index import index_policy
from relocation_rag.retrieval import retrieve_policy

QUESTION = ("My approved move was completed 45 days ago in 2026. "
            "Under the current policy, am I still within the filing window?")


def main():
    corpus = Path("submission/handbooks").resolve()
    current = corpus / "rel-103-claims-2026.md"
    archive = corpus / "rel-103-claims-2025.md"
    heading, body = current.read_text().split("## ", 1)
    old_body = archive.read_text().split("## ", 1)[1]
    with TemporaryDirectory() as directory:
        os.chdir(directory)
        for path in sorted(corpus.glob("*.md")):
            index_policy(path)
        staged = Path(current.name)
        for label, content in [("Stale copy published as current", old_body),
                               ("Approved edition restored", body)]:
            staged.write_text(heading + "## " + content)
            index_policy(staged)
            results = retrieve_policy(QUESTION)
            print(label, "|", QUESTION)
            for key, passage in zip(results["ids"][0], results["documents"][0], strict=True):
                if "/REL-103.3/" in key and "Submit the claim within" in passage:
                    print("Retrieved:", key)
                    print("Source:", next(s for s in passage.split(". ")
                                          if s.startswith("Submit the claim within")))
                    break
            print(generate_answer(QUESTION))


if __name__ == "__main__":
    main()

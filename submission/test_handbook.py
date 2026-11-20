import re
import pytest

from relocation_rag.generate_answer import generate_answer
from relocation_rag.retrieval import retrieve_policy
from submission.cases import CASES


@pytest.mark.parametrize("question,section,pattern", CASES)
def test_handbook(question, section, pattern):
    results = retrieve_policy(question)
    ids = results["ids"][0][:3]
    hit = any(f"/REL-{section}/" in key for key in ids)
    answer = generate_answer(question)
    # Evaluate generated prose independently of the appended source listing.
    prose = answer.split("Sources:")[0]
    correct = bool(re.search(pattern, prose, re.I))
    print(f"\n{section} | recall@3={int(hit)} | key-info={int(correct)} | {question}")
    print(answer)
    assert hit and correct

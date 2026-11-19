"""Run explicitly with pytest evaluation against the indexed policies."""

import pytest

from relocation_rag.generate_answer import generate_answer
from relocation_rag.retrieval import retrieve_policy

from .cases import CASES
from .judge import judge_answer
from .models import PolicyCase

pytestmark = pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])


def test_recall_at_ten(case: PolicyCase) -> None:
    retrieved = set(retrieve_policy(case.question)["ids"][0][:10])
    missing = [group for group in case.evidence if retrieved.isdisjoint(group)]
    required = len(case.evidence)
    print(f"{case.name} recall@10 = {required - len(missing)}/{required}")
    assert len(missing) < required, f"No required evidence in top ten: {case.evidence}"


def test_answer_key_information(case: PolicyCase) -> None:
    answer = generate_answer(case.question)
    verdicts = judge_answer(case, answer)
    failures = [
        f"{criterion} — {result['reason']}"
        for criterion, result in zip(case.rubric, verdicts, strict=True)
        if result["verdict"] == "no"
    ]
    assert not failures, "\n".join([*failures, f"Generated answer: {answer}"])

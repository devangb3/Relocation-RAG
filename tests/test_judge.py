import json
from dataclasses import replace
from unittest.mock import Mock

import pytest

from evaluation.cases import CASES
from evaluation.judge import FoundryClient, judge_answer


@pytest.fixture
def judge(monkeypatch):
    client = Mock(spec=FoundryClient)
    monkeypatch.setattr(FoundryClient, "from_env", lambda: client)
    return client.respond


def test_judge_maps_each_verdict_to_the_requested_criterion(judge):
    case = next(case for case in CASES if case.name == "corrected-folio")
    case = replace(case, rubric=("Was the claim timely?", "Was it submitted on February 20?"))
    answer = "The employee submitted the claim on February 20."
    missing = {"verdict": "no", "reason": "The answer omits whether the claim was timely."}
    present = {"verdict": "yes", "reason": answer}
    judge.return_value = json.dumps({"2": present, "1": missing})
    assert judge_answer(case, answer) == [missing, present]
    judge.assert_called_once()
    prompt = json.loads(judge.call_args.args[1])
    assert prompt["candidate_answer"] == answer
    assert prompt["reference_answer"] == case.expected_answer
    assert list(prompt["criteria"].values()) == list(case.rubric)
    assert judge.call_args.kwargs["text_format"]["schema"]["required"] == ["1", "2"]


@pytest.mark.parametrize(
    "response",
    [
        "bad json",
        "{}",
        '{"1": {"verdict": "maybe", "reason": "unsure"}}',
        '{"1": {"verdict": "yes", "reason": ""}}',
    ],
)
def test_malformed_judgment_is_an_error(judge, response):
    case = next(case for case in CASES if case.name == "packing")
    judge.return_value = response
    with pytest.raises(ValueError, match="Judge returned"):
        judge_answer(case, "The claimable amount is $410.")

from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyCase:
    name: str
    question: str
    expected_answer: str
    # One hit per group. Equivalent chunks from different editions share a group.
    evidence: tuple[tuple[str, ...], ...]
    rubric: tuple[str, ...]

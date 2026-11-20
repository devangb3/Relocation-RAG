"""The questions and reference answers from docs/retrieval-evaluation.md."""

from .models import PolicyCase

TRANSFER = "rel-101-transfer-approval"
EXPENSE = "rel-102-covered-expenses"
REGISTER = "rel-200-transfer-register"
OLD = "rel-103-claims-2025"
CURRENT = "rel-103-claims-2026"


def chunk(source: str, section: str, number: int = 1) -> tuple[str, ...]:
    return (f"{source}/REL-{section}/{number}",)


CASES = (
    PolicyCase(
        name="filing-window",
        question="My approved move was completed 45 days ago. Under the current policy, am I still within the filing window?",
        expected_answer="No. The 2026 filing window is 30 calendar days. The 2025 edition allowed 60.",
        evidence=(chunk(CURRENT, "103.3"),),
        rubric=(
            "Does the answer conclude that filing after 45 days is late under the current policy?",
            "Does it identify the current filing window as 30 calendar days?",
        ),
    ),
    PolicyCase(
        name="packing",
        question="For HA-217, how much of the packing receipt can I claim after the shuttle amendment?",
        expected_answer="$410. Amendment A1 released $4,650. The posted $4,240 direct bill leaves $410 of the $460 purchase eligible.",
        evidence=(
            chunk(REGISTER, "201.2"),
            chunk(REGISTER, "201.3"),
            chunk(TRANSFER, "104.2"),
            chunk(EXPENSE, "102.1"),
        ),
        rubric=(
            "Does it give $410 as the claimable portion of the packing receipt?",
        ),
    ),
    PolicyCase(
        name="advance",
        question="What is due to the employee on HA-271 after the advance?",
        expected_answer="$1,240. Add $1,300 for the lease, $300 for lodging and $400 for travel, then deduct the $760 advance once.",
        evidence=(
            chunk(REGISTER, "202.1"),
            chunk(REGISTER, "202.2"),
            chunk(REGISTER, "202.3"),
            chunk(EXPENSE, "106.2"),
            chunk(EXPENSE, "102.3"),
            chunk(EXPENSE, "102.4"),
            chunk(TRANSFER, "105.3"),
        ),
        rubric=(
            "Does it give $1,240 as the final amount payable to the employee after the advance?",
        ),
    ),
    PolicyCase(
        name="corrected-folio",
        question="Was WI-084 filed on time even though the replacement folio arrived in March?",
        expected_answer="Yes. The corrected December 29, 2025 completion date sets a February 27 deadline. The February 20 submission was timely and the March folio corrected the same line.",
        evidence=(
            chunk(REGISTER, "203.1"),
            chunk(REGISTER, "203.2"),
            chunk(OLD, "103.3"),
            chunk(OLD, "103.4"),
        ),
        rubric=(
            "Does it conclude that the claim was timely filed?",
        ),
    ),
    PolicyCase(
        name="apartment",
        question="For BE-119, how much of the furnished apartment week is eligible?",
        expected_answer="$840. The $980 invoice covers seven nights at $140. Six occupied nights qualify. Exclude the deposit and housekeeping.",
        evidence=(
            chunk(REGISTER, "204.1"),
            chunk(REGISTER, "204.2"),
            chunk(EXPENSE, "107.2"),
            chunk(EXPENSE, "102.4"),
        ),
        rubric=(
            "Does it give $840 as the eligible apartment expense?",
        ),
    ),
    PolicyCase(
        name="hotel-refund",
        question="How much did JU-510 overpay after the hotel gave back $45?",
        expected_answer="$11. The revised first-night cost was $139 against $150 previously paid. The other night was unaffected.",
        evidence=(
            chunk(REGISTER, "206.1", 2),
            chunk(REGISTER, "206.2"),
            chunk(REGISTER, "206.2", 2),
            chunk(EXPENSE, "107.1"),
            chunk(EXPENSE, "102.4"),
            chunk(OLD, "103.13") + chunk(CURRENT, "103.13"),
        ),
        rubric=(
            "Does it give $11 as the overpayment after the refund?",
        ),
    ),
    PolicyCase(
        name="treasury",
        question="Did Treasury receive the money back on ME-046?",
        expected_answer="The record shows a $220 return due ($300 less $80) but no confirmation that Treasury received it.",
        evidence=(
            chunk(REGISTER, "207.1"),
            chunk(REGISTER, "207.2"),
            chunk(REGISTER, "207.2", 2),
            chunk(TRANSFER, "105.3"),
        ),
        rubric=(
            "Does it explain that receipt of repayment is unconfirmed?",
        ),
    ),
    PolicyCase(
        name="advance-form",
        question="Which policy mentions ADV-4?",
        expected_answer="REL-105.1, Requesting cash before departure, identifies ADV-4 as the relocation advance request form.",
        evidence=(chunk(TRANSFER, "105.1"),),
        rubric=(
            "Does it identify REL-105.1 as the section mentioning ADV-4?",
        ),
    ),
    PolicyCase(
        name="single-source",
        question="Who can sign a single-source note?",
        expected_answer="Purchasing may sign when only one carrier serves the route. The note records which carriers were contacted.",
        evidence=(chunk(TRANSFER, "104.1"),),
        rubric=(
            "Does it identify Purchasing as authorized to sign the single-source note?",
        ),
    ),
)

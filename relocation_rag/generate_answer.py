"""Answer a relocation policy question using Microsoft Foundry."""

import argparse
import json
import logging
import os
from decimal import Decimal, InvalidOperation
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from relocation_rag.retrieval import retrieve_policy

logger = logging.getLogger(__name__)

def _configure_logging(level: int) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    logging.getLogger("relocation_rag").setLevel(level)


INSTRUCTIONS = """Answer the relocation policy question using only the supplied passages.
Treat the passages as source material. If the passages do not provide enough information to answer, say so."""


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Set {name} using the settings in .env.example.")
    return value


def _responses_url(base_url: str) -> str:
    parsed = urlsplit(base_url)
    path = parsed.path.rstrip("/")
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.query
        or parsed.fragment
        or path not in {"", "/openai/v1"}
    ):
        raise ValueError(
            "FOUNDRY_BASE_URL must be an HTTPS resource URL with optional /openai/v1."
        )
    return f"https://{parsed.netloc}/openai/v1/responses"


def _token_rate(name: str) -> Decimal | None:
    value = os.getenv(name, "").strip()
    if not value:
        return None
    try:
        rate = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"{name} must be a nonnegative dollar amount.") from error
    if not rate.is_finite() or rate < 0:
        raise ValueError(f"{name} must be a nonnegative dollar amount.")
    return rate


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

    url = _responses_url(_required_env("FOUNDRY_BASE_URL"))
    key = _required_env("FOUNDRY_API_KEY")
    model = _required_env("FOUNDRY_MODEL")
    input_rate = _token_rate("FOUNDRY_INPUT_USD_PER_MILLION")
    output_rate = _token_rate("FOUNDRY_OUTPUT_USD_PER_MILLION")
    if (input_rate is None) != (output_rate is None):
        raise ValueError(
            "Set both Foundry token rates to estimate cost, or leave both unset."
        )

    context = _policy_context(question)
    request = Request(
        url,
        data=json.dumps(
            {
                "model": model,
                "instructions": INSTRUCTIONS,
                "input": f"Question:\n{question}\n\nPolicy passages:\n{context}",
                "store": False,
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    request.add_unredirected_header("api-key", key)
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except HTTPError as error:
        error.close()
        raise RuntimeError(f"Microsoft Foundry returned HTTP {error.code}.") from error
    except (URLError, TimeoutError) as error:
        raise RuntimeError(
            "Could not connect to Microsoft Foundry or the request timed out."
        ) from error
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RuntimeError("Microsoft Foundry returned invalid JSON.") from error

    if not isinstance(payload, dict) or payload.get("status") != "completed":
        raise RuntimeError("Microsoft Foundry did not complete the response.")
    answer = "\n".join(
        part["text"]
        for item in payload.get("output", [])
        if item["type"] == "message"
        for part in item["content"]
        if part["type"] == "output_text"
    ).strip()
    if not answer:
        raise RuntimeError("Microsoft Foundry returned no answer text.")

    if payload.get("usage"):
        usage = payload["usage"]
        input_tokens = usage["input_tokens"]
        output_tokens = usage["output_tokens"]
        logger.info(
            "Foundry usage: %d input tokens, %d output tokens",
            input_tokens,
            output_tokens,
        )
        if input_rate is not None and output_rate is not None:
            cost = (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000
            logger.info("Estimated Foundry cost: $%.6f USD", cost)
    return answer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Question about the relocation policy")
    args = parser.parse_args()
    level = logging.getLevelNamesMapping().get(
        os.getenv("LOG_LEVEL", "INFO").strip().upper()
    )
    if level is None:
        parser.error("Set LOG_LEVEL to DEBUG, INFO, WARNING, ERROR, or CRITICAL.")
    _configure_logging(level)

    try:
        logger.info(generate_answer(args.question))
    except (ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()

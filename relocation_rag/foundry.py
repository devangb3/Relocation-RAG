"""Shared Microsoft Foundry Responses API transport."""

import json
import logging
import os
import urllib.request
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request

logger = logging.getLogger(__name__)


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


@dataclass(frozen=True)
class FoundryClient:
    url: str
    key: str
    model: str
    input_rate: Decimal | None
    output_rate: Decimal | None

    @classmethod
    def from_env(cls):
        url = _responses_url(_required_env("FOUNDRY_BASE_URL"))
        key = _required_env("FOUNDRY_API_KEY")
        model = _required_env("FOUNDRY_MODEL")
        input_rate = _token_rate("FOUNDRY_INPUT_USD_PER_MILLION")
        output_rate = _token_rate("FOUNDRY_OUTPUT_USD_PER_MILLION")
        if (input_rate is None) != (output_rate is None):
            raise ValueError(
                "Set both Foundry token rates to estimate cost, or leave both unset."
            )

        return cls(url, key, model, input_rate, output_rate)

    def respond(
        self, instructions: str, prompt: str, *, text_format: dict | None = None
    ) -> str:
        body = {
            "model": self.model,
            "instructions": instructions,
            "input": prompt,
            "store": False,
        }
        if text_format is not None:
            body["text"] = {"format": text_format}
        request = Request(
            self.url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        request.add_unredirected_header("api-key", self.key)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.load(response)
        except HTTPError as error:
            error.close()
            raise RuntimeError(
                f"Microsoft Foundry returned HTTP {error.code}."
            ) from error
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
            if self.input_rate is not None and self.output_rate is not None:
                cost = (
                    input_tokens * self.input_rate + output_tokens * self.output_rate
                ) / 1_000_000
                logger.info("Estimated Foundry cost: $%.6f USD", cost)
        return answer

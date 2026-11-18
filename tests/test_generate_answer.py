import json
from io import BytesIO
from urllib.error import HTTPError

import pytest

from relocation_rag import generate_answer


@pytest.fixture
def foundry_settings(monkeypatch):
    monkeypatch.setenv("FOUNDRY_BASE_URL", "https://example.openai.azure.com")
    monkeypatch.setenv("FOUNDRY_API_KEY", "test-key")
    monkeypatch.setenv("FOUNDRY_MODEL", "test-deployment")
    monkeypatch.setenv("FOUNDRY_INPUT_USD_PER_MILLION", "0.15")
    monkeypatch.setenv("FOUNDRY_OUTPUT_USD_PER_MILLION", "0.60")

    def retrieve_policy(question):
        assert question == "Can I file after 45 days?"
        return {
            "documents": [["Submit the claim within 30 calendar days."]],
            "metadatas": [
                [
                    {
                        "section": "REL-103.3",
                        "source": "rel-103-claims-2026.md",
                        "effective_date": "2026-01-01",
                    }
                ]
            ],
        }

    monkeypatch.setattr(generate_answer, "retrieve_policy", retrieve_policy)


@pytest.mark.parametrize("base_path", ["", "/openai/v1/"])
def test_generates_cited_answer_and_logs_usage(
    foundry_settings, monkeypatch, caplog, base_path
):
    monkeypatch.setenv(
        "FOUNDRY_BASE_URL", f"https://example.openai.azure.com{base_path}"
    )
    citation = "[REL-103.3 | rel-103-claims-2026.md | effective 2026-01-01]"

    def urlopen(request, *, timeout):
        assert (
            request.full_url == "https://example.openai.azure.com/openai/v1/responses"
        )
        assert request.get_method() == "POST"
        assert request.get_header("Api-key") == "test-key"
        assert timeout > 0
        body = json.loads(request.data)
        assert body["model"] == "test-deployment"
        assert body["store"] is False
        assert "Can I file after 45 days?" in body["input"]
        assert f"{citation}\nSubmit the claim within 30 calendar days." in body["input"]
        return BytesIO(
            json.dumps(
                {
                    "status": "completed",
                    "output": [
                        {"type": "reasoning"},
                        {
                            "type": "message",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "The filing window is 30 calendar days.",
                                },
                                {"type": "output_text", "text": citation},
                            ],
                        },
                    ],
                    "usage": {"input_tokens": 1000, "output_tokens": 200},
                }
            ).encode()
        )

    monkeypatch.setattr(generate_answer, "urlopen", urlopen)
    with caplog.at_level("INFO"):
        answer = generate_answer.generate_answer("Can I file after 45 days?")

    assert answer == f"The filing window is 30 calendar days.\n{citation}"
    assert "1000 input tokens, 200 output tokens" in caplog.text
    assert "$0.000270 USD" in caplog.text


def test_empty_index_stops_before_foundry_request(foundry_settings, monkeypatch):
    monkeypatch.setattr(
        generate_answer, "retrieve_policy", lambda _: {"documents": [[]]}
    )

    def urlopen(*args, **kwargs):
        pytest.fail("Foundry was called without policy passages")

    monkeypatch.setattr(generate_answer, "urlopen", urlopen)
    with pytest.raises(ValueError, match="No policy passages found"):
        generate_answer.generate_answer("Can I file after 45 days?")


def test_foundry_http_error_omits_response_body(foundry_settings, monkeypatch):
    def urlopen(request, *, timeout):
        raise HTTPError(
            request.full_url, 401, "Unauthorized", None, BytesIO(b"private details")
        )

    monkeypatch.setattr(generate_answer, "urlopen", urlopen)
    with pytest.raises(RuntimeError, match=r"^Microsoft Foundry returned HTTP 401\.$"):
        generate_answer.generate_answer("Can I file after 45 days?")

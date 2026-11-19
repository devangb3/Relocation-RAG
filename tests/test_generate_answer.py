import json
import logging
import runpy
import sys
import urllib.request
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

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    with caplog.at_level("INFO"):
        answer = generate_answer.generate_answer("Can I file after 45 days?")

    assert answer == f"The filing window is 30 calendar days.\n{citation}"
    assert "1000 input tokens, 200 output tokens" in caplog.text
    assert "$0.000270 USD" in caplog.text


@pytest.mark.parametrize("log_level,expect_usage", [("INFO", True), ("WARNING", False)])
def test_cli_prints_answer_to_stdout(
    foundry_settings, monkeypatch, caplog, capsys, log_level, expect_usage
):
    monkeypatch.setenv("LOG_LEVEL", log_level)
    monkeypatch.setattr(
        sys, "argv", ["relocation_rag.generate_answer", "Can I file after 45 days?"]
    )

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

    monkeypatch.setattr("relocation_rag.retrieval.retrieve_policy", retrieve_policy)

    def urlopen(request, *, timeout):
        assert timeout > 0
        return BytesIO(
            json.dumps(
                {
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "The filing window is 30 calendar days.",
                                }
                            ],
                        }
                    ],
                    "usage": {"input_tokens": 1000, "output_tokens": 200},
                }
            ).encode()
        )

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    root_logger = logging.getLogger()
    package_logger = logging.getLogger("relocation_rag")
    previous = (root_logger.level, package_logger.level, list(root_logger.handlers))
    try:
        runpy.run_path(generate_answer.__file__, run_name="__main__")
    finally:
        root_logger.setLevel(previous[0])
        package_logger.setLevel(previous[1])
        root_logger.handlers[:] = previous[2]

    captured = capsys.readouterr()
    answer = "The filing window is 30 calendar days."
    assert captured.out == f"{answer}\n"
    assert answer not in captured.err
    assert answer not in caplog.text
    if expect_usage:
        assert "1000 input tokens, 200 output tokens" in caplog.text
        assert "1000 input tokens, 200 output tokens" in captured.err
    else:
        assert "input tokens" not in caplog.text
        assert "input tokens" not in captured.err


def test_empty_index_stops_before_foundry_request(foundry_settings, monkeypatch):
    monkeypatch.setattr(
        generate_answer, "retrieve_policy", lambda _: {"documents": [[]]}
    )

    def urlopen(*args, **kwargs):
        pytest.fail("Foundry was called without policy passages")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    with pytest.raises(ValueError, match="No policy passages found"):
        generate_answer.generate_answer("Can I file after 45 days?")


def test_foundry_http_error_omits_response_body(foundry_settings, monkeypatch):
    def urlopen(request, *, timeout):
        raise HTTPError(
            request.full_url, 401, "Unauthorized", None, BytesIO(b"private details")
        )

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    with pytest.raises(RuntimeError, match=r"^Microsoft Foundry returned HTTP 401\.$"):
        generate_answer.generate_answer("Can I file after 45 days?")


@pytest.mark.parametrize(("setting", "value", "error"), [
    ("FOUNDRY_MODEL", " ", "Set FOUNDRY_MODEL"),
    ("FOUNDRY_BASE_URL", "http://example.org", "HTTPS resource URL"),
    ("FOUNDRY_INPUT_USD_PER_MILLION", "", "Set both Foundry token rates"),
    ("FOUNDRY_INPUT_USD_PER_MILLION", "abc", "nonnegative dollar amount"),
    ("FOUNDRY_OUTPUT_USD_PER_MILLION", "-1", "nonnegative dollar amount"),
    ("FOUNDRY_OUTPUT_USD_PER_MILLION", "NaN", "nonnegative dollar amount"),
])
def test_bad_settings_fail_before_retrieval(foundry_settings, monkeypatch, setting, value, error):
    monkeypatch.setenv(setting, value)
    monkeypatch.setattr(generate_answer, "retrieve_policy", lambda _: pytest.fail("retrieval ran"))
    with pytest.raises(ValueError, match=error):
        generate_answer.generate_answer("Can I file after 45 days?")


@pytest.mark.parametrize(("response", "error"), [
    (b"Bad gateway", "invalid JSON"),
    (b'{"status":"incomplete"}', "did not complete"),
    (b'{"status":"completed","output":[]}', "no answer text"),
])
def test_unusable_foundry_response(foundry_settings, monkeypatch, response, error):
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: BytesIO(response))
    with pytest.raises(RuntimeError, match=error):
        generate_answer.generate_answer("Can I file after 45 days?")


def test_foundry_timeout(foundry_settings, monkeypatch):
    def timeout(request, *, timeout):
        raise TimeoutError()

    monkeypatch.setattr(urllib.request, "urlopen", timeout)
    with pytest.raises(RuntimeError, match="request timed out"):
        generate_answer.generate_answer("Can I file after 45 days?")


def test_missing_citations_stop_answer_generation(foundry_settings, monkeypatch):
    passages = {"documents": [["Keep receipts."]], "metadatas": None}
    monkeypatch.setattr(generate_answer, "retrieve_policy", lambda _: passages)
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: pytest.fail("Foundry called"))
    with pytest.raises(ValueError, match="missing citation metadata"):
        generate_answer.generate_answer("Can I file after 45 days?")


def test_search_loop_returns_evidence_and_stops_at_budget(foundry_settings, monkeypatch):
    queries = []
    requests = []

    def context(query):
        queries.append(query)
        return f"Evidence for {query}"

    def urlopen(request, *, timeout):
        body = json.loads(request.data)
        requests.append(body)
        turn = len(requests)
        if turn > 1:
            assert body["input"][-1] == {
                "type": "function_call_output",
                "call_id": f"call-{turn - 1}",
                "output": f"Evidence for policy {turn - 1}",
            }
            assert body["input"][-3]["encrypted_content"] == "reasoning state"
        if turn <= generate_answer.MAX_FOLLOWUP_SEARCHES:
            assert body["tool_choice"] == "auto"
            output = [
                {"type": "reasoning", "encrypted_content": "reasoning state"},
                {"type": "function_call", "name": "search_corpus",
                 "arguments": json.dumps({"query": f"policy {turn}"}),
                 "call_id": f"call-{turn}"},
            ]
        else:
            assert body["tool_choice"] == "none"
            output = [{"type": "message", "content": [{"type": "output_text", "text": "Done"}]}]
        return BytesIO(json.dumps({"status": "completed", "output": output}).encode())

    monkeypatch.setattr(generate_answer, "_policy_context", context)
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    assert generate_answer.generate_answer("case question") == "Done"
    assert queries == ["case question", "policy 1", "policy 2", "policy 3"]

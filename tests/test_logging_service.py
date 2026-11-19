import logging
import os
import warnings

import pytest
from huggingface_hub.utils import are_progress_bars_disabled
from transformers.utils.logging import is_progress_bar_enabled

from relocation_rag.logging_service import LoggingService

_LIBRARIES = ("huggingface_hub", "sentence_transformers", "transformers")


@pytest.fixture
def restore_logging():
    root = logging.getLogger()
    package = logging.getLogger("relocation_rag")
    previous_class = logging.getLoggerClass()
    previous_filters = warnings.filters[:]
    previous_root = (root.level, list(root.handlers))
    previous_package = (package.level, package.propagate, package.disabled)
    snapshot = [
        (logger, logger.level, logger.disabled, logger.propagate, list(logger.handlers))
        for logger in logging.Logger.manager.loggerDict.values()
        if isinstance(logger, logging.Logger)
    ]
    yield
    logging.setLoggerClass(previous_class)
    warnings.filters[:] = previous_filters
    root.setLevel(previous_root[0])
    root.handlers[:] = previous_root[1]
    package.setLevel(previous_package[0])
    package.propagate = previous_package[1]
    package.disabled = previous_package[2]
    for logger, level, disabled, propagate, handlers in snapshot:
        logger.setLevel(level)
        logger.disabled = disabled
        logger.propagate = propagate
        logger.handlers[:] = handlers


def test_index_messages_stay_visible(monkeypatch, capsys, restore_logging):
    from relocation_rag.index import logger as index_logger

    monkeypatch.setenv("LOG_LEVEL", "INFO")
    LoggingService()
    assert index_logger.name == "relocation_rag.index"
    logging.getLogger("__main__").info("Indexed all policies.")
    index_logger.info("Indexed all policies.")

    captured = capsys.readouterr()
    assert captured.err.count("Indexed all policies.") == 1
    assert "INFO Indexed all policies." in captured.err


def test_rejects_unknown_log_level(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "VERBOSE")
    with pytest.raises(ValueError, match="LOG_LEVEL"):
        LoggingService()


def test_package_level_does_not_open_library_info(monkeypatch, restore_logging):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    LoggingService()

    assert logging.getLogger("relocation_rag").level == logging.DEBUG
    assert logging.getLogger().level == logging.WARNING
    library = logging.getLogger("sentence_transformers")
    assert library.level != logging.ERROR
    assert library.getEffectiveLevel() >= logging.WARNING


def test_initialization_hides_foreign_logs(monkeypatch, capsys, restore_logging):
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    LoggingService()
    logging.getLogger("huggingface_hub.utils._http").warning(
        "Warning: You are sending unauthenticated requests to the HF Hub."
    )
    logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").warning(
        "NOTE: Redirects are currently not supported in MacOs."
    )
    logging.getLogger("some.other.package").error("library failure")
    logging.getLogger("relocation_rag.generate_answer").info(
        "Foundry usage: 1 input tokens, 1 output tokens"
    )

    captured = capsys.readouterr()
    assert "unauthenticated" not in captured.err
    assert "Redirects are currently not supported" not in captured.err
    assert "library failure" not in captured.err
    assert "Foundry usage: 1 input tokens, 1 output tokens" in captured.err


def test_initialization_disables_download_bars_without_changing_library_levels(
    monkeypatch, restore_logging
):
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    libraries = [logging.getLogger(name) for name in _LIBRARIES]
    levels_before = [logger.level for logger in libraries]

    LoggingService()

    assert [logger.level for logger in libraries] == levels_before
    assert all(logger.level != logging.ERROR for logger in libraries)
    assert is_progress_bar_enabled() is False
    assert are_progress_bars_disabled() is True
    assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"

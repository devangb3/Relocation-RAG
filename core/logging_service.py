"""Package diagnostics on stderr. Logs from other packages stay hidden."""

import logging
import os
import sys
import warnings

_PACKAGE_LOGGER = "core"
_LOG_FORMAT = "%(levelname)s %(message)s"
_LEVEL_NAMES = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


def _is_package(name: str) -> bool:
    return name == _PACKAGE_LOGGER or name.startswith(f"{_PACKAGE_LOGGER}.")


class _PackageOnlyFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return _is_package(record.name)


class _QuietLogger(logging.Logger):
    """Logger that stays silent unless it belongs to this package."""

    def __init__(self, name: str, level: int = logging.NOTSET) -> None:
        super().__init__(name, level)
        if not _is_package(name):
            self.disabled = True
            self.propagate = False


class LoggingService:
    """Send core diagnostics to stderr and hide every other logger.

    Libraries such as Torch and Hugging Face attach their own handlers, and
    some of them log while they are imported. Construction installs a quiet
    logger class first, then drops handlers already attached to foreign
    loggers. Download bars are turned off separately because they write to
    stderr without going through logging.
    """

    def __init__(self, level_name: str | None = None) -> None:
        raw = os.getenv("LOG_LEVEL", "INFO") if level_name is None else level_name
        normalized = raw.strip().upper()
        level = logging.getLevelNamesMapping().get(normalized)
        if normalized not in _LEVEL_NAMES or level is None:
            raise ValueError(
                "Set LOG_LEVEL to DEBUG, INFO, WARNING, ERROR, or CRITICAL."
            )
        self.level_name = normalized
        self.level = level
        self._configure_package_logger()
        self._silence_foreign_loggers()
        self._disable_download_progress()

    def _configure_package_logger(self) -> None:
        root = logging.getLogger()
        if not any(isinstance(handler, _StderrHandler) for handler in root.handlers):
            root.addHandler(_StderrHandler())
        root.setLevel(logging.WARNING)

        package = logging.getLogger(_PACKAGE_LOGGER)
        package.disabled = False
        package.setLevel(self.level)
        package.propagate = True

    def _silence_foreign_loggers(self) -> None:
        logging.setLoggerClass(_QuietLogger)
        warnings.simplefilter("ignore")
        for name, logger in list(logging.Logger.manager.loggerDict.items()):
            if not isinstance(logger, logging.Logger):
                continue
            if _is_package(name):
                logger.disabled = False
                continue
            logger.disabled = True
            logger.propagate = False
            for handler in list(logger.handlers):
                logger.removeHandler(handler)

    def _disable_download_progress(self) -> None:
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        if "huggingface_hub" not in sys.modules and "transformers" not in sys.modules:
            return

        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
        if "transformers" in sys.modules:
            from transformers.utils.logging import disable_progress_bar

            disable_progress_bar()


class _StderrHandler(logging.StreamHandler):
    def __init__(self) -> None:
        super().__init__(sys.stderr)
        self.setFormatter(logging.Formatter(_LOG_FORMAT))
        self.addFilter(_PackageOnlyFilter())

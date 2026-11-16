"""Logging setup: human-readable by default, one JSON object per line with --log-json."""
from __future__ import annotations

import json
import logging
import sys
import time

_RESERVED = set(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
                  + f".{int(record.msecs):03d}Z",
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for k, v in record.__dict__.items():
            if k not in _RESERVED and not k.startswith("_"):
                out[k] = v
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


class TextFormatter(logging.Formatter):
    """`[time] LEVEL logger: message key=value ...`"""

    def __init__(self):
        super().__init__("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extra = " ".join(f"{k}={_one_line(v)}" for k, v in record.__dict__.items()
                         if k not in _RESERVED and not k.startswith("_"))
        return _one_line(f"{base} {extra}" if extra else base) if not record.exc_info else (
            f"{base} {extra}" if extra else base)


def _one_line(v) -> str:
    """Escape line breaks so a value can never forge an extra log line."""
    return str(v).replace("\r", "\\r").replace("\n", "\\n")


def setup_logging(level: str = "INFO", json_logs: bool = False) -> None:
    root = logging.getLogger("freewidth")
    root.handlers.clear()
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(JsonFormatter() if json_logs else TextFormatter())
    root.addHandler(h)
    root.setLevel(level.upper())
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"freewidth.{name}")

from __future__ import annotations

from .base import Backend, GenResult


def make_backend(cfg: dict) -> Backend:
    kind = cfg.get("backend", "hf")
    kw = {k: v for k, v in cfg.items() if k != "backend"}
    if kind == "hf":
        from .hf import HFBackend

        return HFBackend(**kw)
    if kind in ("openai", "openai_compat", "vllm", "api"):
        from .openai_compat import OpenAICompatBackend

        return OpenAICompatBackend(**kw)
    if kind == "mock":
        from .mock import MockBackend

        return MockBackend(**kw)
    raise ValueError(f"unknown backend {kind!r}")


__all__ = ["Backend", "GenResult", "make_backend"]

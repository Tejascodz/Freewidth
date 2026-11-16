"""Security properties outside the HTTP server."""
import logging

import pytest

from freewidth.backends.openai_compat import _check_base_url
from freewidth.logs import TextFormatter


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com", "gopher://x", "http://", "nonsense"])
def test_remote_backend_rejects_non_http_urls(url):
    with pytest.raises(ValueError):
        _check_base_url(url, sends_key=False, allow_insecure_http=False)


def test_remote_backend_never_sends_key_in_clear_to_remote_host():
    with pytest.raises(ValueError):
        _check_base_url("http://api.example.com/v1", sends_key=True, allow_insecure_http=False)
    assert _check_base_url("http://localhost:8000/v1/", sends_key=True, allow_insecure_http=False) \
        == "http://localhost:8000/v1"
    assert _check_base_url("https://api.example.com/v1", sends_key=True, allow_insecure_http=False)
    assert _check_base_url("http://10.0.0.2/v1", sends_key=True, allow_insecure_http=True)


def test_text_logs_cannot_be_forged_with_newlines():
    rec = logging.LogRecord("freewidth.x", logging.INFO, __file__, 1, "msg\nFAKE LINE", (), None)
    rec.request_id = "a\nERROR forged"
    out = TextFormatter().format(rec)
    assert "\n" not in out and "\\n" in out


def test_hf_backend_does_not_trust_remote_code_by_default():
    import inspect

    from freewidth.backends.hf import HFBackend

    assert inspect.signature(HFBackend.__init__).parameters["trust_remote_code"].default is False


def test_configs_do_not_enable_remote_code_or_open_binds():
    import glob

    import yaml

    for p in glob.glob("configs/*.yaml"):
        with open(p) as f:
            text = f.read()
        assert "trust_remote_code: true" not in text.lower(), p
        data = yaml.safe_load(text) or {}
        server = (data.get("engine") or {}).get("server") or {}
        assert not server.get("allow_unauthenticated"), p


def test_hf_seed_touches_only_its_own_gpu():
    """Concurrent waves on different GPUs must not reseed each other's generator."""
    import contextlib
    from types import SimpleNamespace

    from freewidth.backends.hf import HFBackend

    calls = []
    current = {"dev": None}

    @contextlib.contextmanager
    def device_ctx(d):
        prev, current["dev"] = current["dev"], d
        yield
        current["dev"] = prev

    fake = SimpleNamespace(
        device=lambda s: s,
        manual_seed=lambda s: calls.append(("ALL", s)),
        cuda=SimpleNamespace(is_available=lambda: True, device=device_ctx,
                             manual_seed=lambda s: calls.append((current["dev"], s))),
        mps=SimpleNamespace(manual_seed=lambda s: calls.append(("mps", s))),
    )
    b = HFBackend.__new__(HFBackend)
    b._torch, b.device = fake, "cuda:1"
    b._seed(7)
    b.device = "cuda:0"
    b._seed(9)
    assert calls == [("cuda:1", 7), ("cuda:0", 9)]      # never a process-wide reseed

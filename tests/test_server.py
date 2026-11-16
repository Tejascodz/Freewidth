"""OpenAI-compatible server on the simulated ladder."""
import os

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from freewidth.engine import Engine, EngineConfig  # noqa: E402


@pytest.fixture()
def engine(tmp_path):
    cfg = EngineConfig.model_validate({
        "ladder": [{"tag": "s", "backend": "mock", "params_b": 0.8},
                   {"tag": "b", "backend": "mock", "params_b": 4.0}],
        "sampling": {"max_new_tokens": 512},
        "calibration": {"dir": str(tmp_path / "calib")},
    })
    e = Engine(cfg)
    e.load()
    yield e
    e.close()

# ------------------------------------------------------------------ server
@pytest.fixture()
def client(engine, monkeypatch):
    from fastapi.testclient import TestClient

    from freewidth.engine.server import create_app

    monkeypatch.delenv("FREEWIDTH_API_KEY", raising=False)
    return TestClient(create_app(engine))


def test_chat_completion(client):
    r = client.post("/v1/chat/completions", json={
        "model": "freewidth", "messages": [{"role": "system", "content": "be brief"},
                                            {"role": "user", "content": "Compute 2 + 3."}]})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["usage"]["completion_tokens"] > 0
    assert "waves" in body["freewidth"]
    assert r.headers["x-request-id"]


def test_chat_errors_and_stream(client):
    assert client.post("/v1/chat/completions", json={"messages": []}).status_code == 422
    assert client.post("/v1/chat/completions", json={
        "messages": [{"role": "assistant", "content": "hi"}]}).status_code == 400
    assert client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "x"}], "freewidth": {"tau": 9}}).status_code == 400
    assert client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "x"}], "freewidth": {"evil": 1}}).status_code == 400
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Compute 1 + 1."}]}],
        "stream": True, "freewidth": {"name": "single"}})
    assert r.status_code == 200 and r.text.strip().endswith("data: [DONE]")


def test_health_models_metrics_auth(client, monkeypatch):
    assert client.get("/health").json() == {"status": "ok"}      # liveness only, nothing else
    assert len(client.get("/v1/info").json()["ladder"]) == 2
    assert client.get("/v1/models").status_code == 200
    assert "freewidth_queue_depth" in client.get("/metrics").text
    monkeypatch.setenv("FREEWIDTH_API_KEY", "secret")
    ok = {"Authorization": "Bearer secret"}
    for path in ("/v1/models", "/v1/info", "/metrics"):
        assert client.get(path).status_code == 401, path
        assert client.get(path, headers={"Authorization": "Bearer secreT"}).status_code == 401, path
        assert client.get(path, headers=ok).status_code == 200, path
    assert client.get("/health").status_code == 200              # probes keep working
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"


def _post(client, body, **kw):
    return client.post("/v1/chat/completions", json=body, **kw)


def test_request_limits(engine, monkeypatch):
    from fastapi.testclient import TestClient

    from freewidth.engine.server import create_app

    monkeypatch.delenv("FREEWIDTH_API_KEY", raising=False)
    engine.cfg.server.max_prompt_chars = 50
    engine.cfg.server.max_messages = 3
    engine.cfg.server.max_samples = 8
    engine.cfg.server.max_new_tokens = 64
    engine.cfg.server.allowed_policies = ["freewidth", "single"]
    engine.cfg.server.max_body_bytes = 2048
    c = TestClient(create_app(engine))
    u = {"role": "user", "content": "Compute 1 + 1."}
    assert _post(c, {"messages": [{"role": "user", "content": "x" * 60}]}).status_code == 400
    assert _post(c, {"messages": [u] * 4}).status_code == 400
    assert _post(c, {"messages": [u], "freewidth": {"sc_n": 9}}).status_code == 400
    assert _post(c, {"messages": [u], "freewidth": {"width": 64}}).status_code == 400
    assert _post(c, {"messages": [u], "max_tokens": 65}).status_code == 400
    assert _post(c, {"messages": [u], "freewidth": {"name": "big_sc2"}}).status_code == 400
    assert _post(c, {"messages": [u], "seed": -1}).status_code == 422
    assert _post(c, {"messages": [u], "seed": 2**40}).status_code == 422
    big = {"messages": [{"role": "user", "content": "y" * 3000}]}
    assert _post(c, big).status_code == 413
    assert _post(c, {"messages": [u], "freewidth": {"name": "single", "sc_n": 8}}).status_code == 200


def test_request_id_is_sanitised_and_headers_set(client):
    r = client.get("/health", headers={"x-request-id": "abc-123"})
    assert r.headers["x-request-id"] == "abc-123"
    r = client.get("/health", headers={"x-request-id": "bad id\twith spaces"})
    assert r.headers["x-request-id"] != "bad id\twith spaces" and len(r.headers["x-request-id"]) == 16
    assert r.headers["x-content-type-options"] == "nosniff" and r.headers["cache-control"] == "no-store"
    assert client.get("/docs").status_code == 404 and client.get("/openapi.json").status_code == 404


def test_post_without_content_length_is_rejected(client):
    def gen():
        yield b'{"messages": [{"role": "user", "content": "x"}]}'

    r = client.post("/v1/chat/completions", content=gen(), headers={"content-type": "application/json"})
    assert r.status_code == 411


def test_refuses_open_unauthenticated_bind():
    from freewidth.engine.config import ServerConfig
    from freewidth.engine.server import check_exposure, is_loopback

    assert is_loopback("127.0.0.1") and is_loopback("::1") and is_loopback("localhost")
    assert not is_loopback("0.0.0.0") and not is_loopback("10.0.0.5")
    check_exposure(ServerConfig(), key_present=False)                       # loopback default
    with pytest.raises(SystemExit):
        check_exposure(ServerConfig(host="0.0.0.0"), key_present=False)
    check_exposure(ServerConfig(host="0.0.0.0"), key_present=True)
    check_exposure(ServerConfig(host="0.0.0.0", allow_unauthenticated=True), key_present=False)


def test_multi_turn_folding():
    from freewidth.engine.server import ChatMessage, split_messages

    sys_, q = split_messages([ChatMessage(role="user", content="a"), ChatMessage(role="assistant", content="b"),
                              ChatMessage(role="user", content="c")])
    assert sys_ is None and "USER: a" in q and q.endswith("c")


@pytest.mark.skipif(not os.environ.get("FREEWIDTH_GPU_TESTS"), reason="set FREEWIDTH_GPU_TESTS=1 on a GPU box")
def test_real_engine_config_loads():
    cfg = EngineConfig.from_yaml(os.path.join(os.path.dirname(__file__), "..", "configs", "engine_t4.yaml"))
    assert cfg.ladder[0].backend == "hf"

"""OpenAI-compatible HTTP server for the live engine.

    freewidth serve --config configs/engine_t4.yaml

POST /v1/chat/completions   standard body; optional `freewidth` object overrides the policy
                            per request (tau, max_waves, quorum, verify, name, ...)
GET  /v1/models
GET  /health                liveness + loaded ladder, devices, free widths
GET  /metrics               Prometheus text format

GET  /v1/info               engine details (ladder, devices, calibration, policy)

Requests run one at a time (the engine owns the GPU); up to `server.max_queue` wait.

Security model (see SECURITY.md):
- If the env var named by `server.api_key_env` is set, every endpoint except `/health` needs a
  matching Bearer token, compared in constant time. `/metrics` also needs it unless
  `server.public_metrics` is set.
- Without a key the server only starts on a loopback address, unless
  `server.allow_unauthenticated` (or `--insecure-no-auth`) says otherwise.
- Request size, message count, prompt length, sample counts, waves and new tokens are all
  capped by `server.*` limits, so one request cannot claim unbounded GPU time or memory.
- Only `/health` is public, and it reveals nothing but liveness.
"""
from __future__ import annotations

import asyncio
import hmac
import ipaddress
import json
import os
import re
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from ..logs import get_logger
from .core import DeadlineExceeded, Engine

log = get_logger("server")


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: Any = ""


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    top_p: float | None = Field(None, gt=0.0, le=1.0)
    max_tokens: int | None = Field(None, ge=1, le=32768)
    max_completion_tokens: int | None = Field(None, ge=1, le=32768)
    seed: int | None = Field(None, ge=0, le=2**32 - 1)
    stream: bool = False
    freewidth: dict | None = None


_ALLOWED_OVERRIDES = {"name", "tau", "max_waves", "quorum", "min_finished", "width", "max_width",
                      "verify", "verify_accept", "top_c", "sc_n", "free_form_fallback", "raw_prompt",
                      "veto_lo", "accept_hi", "require_verified", "quorum_accept", "stream_verify", "speculative",
                      "trigger_min_finished", "trigger_lead_prob", "runaway_factor", "runaway_min_finished",
                      "verify_wait_s", "flaw_p", "flaw_conf"}


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # OpenAI content parts
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return str(content or "")


def split_messages(messages: list[ChatMessage]) -> tuple[str | None, str]:
    """System prompt + the question. Earlier turns are folded into the question as context."""
    system = "\n\n".join(_text(m.content) for m in messages if m.role in ("system", "developer")) or None
    turns = [m for m in messages if m.role not in ("system", "developer")]
    if not turns or turns[-1].role != "user":
        raise ValueError("the last message must have role 'user'")
    if len(turns) == 1:
        return system, _text(turns[0].content)
    history = "\n\n".join(f"{m.role.upper()}: {_text(m.content)}" for m in turns[:-1])
    return system, f"Conversation so far:\n{history}\n\nCurrent question:\n{_text(turns[-1].content)}"


_RID = re.compile(r"[A-Za-z0-9._-]{1,64}")
_SAMPLE_KEYS = ("sc_n", "width", "max_width")


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def check_exposure(cfg, key_present: bool) -> None:
    """Refuse to serve an unauthenticated engine on a reachable interface by accident."""
    if not key_present and not is_loopback(cfg.host) and not cfg.allow_unauthenticated:
        raise SystemExit(
            f"refusing to listen on {cfg.host} without an API key: set {cfg.api_key_env}, "
            "bind to 127.0.0.1, or pass --insecure-no-auth if this network is trusted")


def enforce_limits(cfg, overrides: dict) -> None:
    """Server-side caps on what one request may ask the engine to do."""
    for k in _SAMPLE_KEYS:
        v = overrides.get(k)
        if isinstance(v, int) and not isinstance(v, bool) and v > cfg.max_samples:
            raise HTTPException(status_code=400, detail=f"{k} exceeds this server's limit ({cfg.max_samples})")
    v = overrides.get("max_waves")
    if isinstance(v, int) and v > cfg.max_waves:
        raise HTTPException(status_code=400, detail=f"max_waves exceeds this server's limit ({cfg.max_waves})")
    v = overrides.get("max_new_tokens")
    if isinstance(v, int) and v > cfg.max_new_tokens:
        raise HTTPException(status_code=400,
                            detail=f"max_tokens exceeds this server's limit ({cfg.max_new_tokens})")
    name = overrides.get("name")
    if name is not None and cfg.allowed_policies is not None and name not in cfg.allowed_policies:
        raise HTTPException(status_code=400, detail=f"policy {name!r} is not enabled on this server")


def create_app(engine: Engine) -> FastAPI:
    cfg = engine.cfg.server
    docs = {} if cfg.expose_docs else {"docs_url": None, "redoc_url": None, "openapi_url": None}
    app = FastAPI(title="FreeWidth", version=_version(), **docs)
    state = {"pending": 0, "started": time.time()}
    gate = asyncio.Lock()

    def _auth(request: Request) -> None:
        key = os.environ.get(cfg.api_key_env)
        if not key:
            return
        got = request.headers.get("authorization", "")
        if not hmac.compare_digest(got.encode(), f"Bearer {key}".encode()):
            raise HTTPException(status_code=401, detail="invalid or missing API key",
                                headers={"WWW-Authenticate": "Bearer"})

    @app.middleware("http")
    async def _request_id(request: Request, call_next):
        got = request.headers.get("x-request-id", "")
        rid = got if _RID.fullmatch(got) else uuid.uuid4().hex[:16]
        request.state.rid = rid
        t0 = time.perf_counter()
        if request.method in ("POST", "PUT", "PATCH"):
            cl = request.headers.get("content-length")
            if cl is None or not cl.isdigit():
                resp = JSONResponse({"error": {"message": "Content-Length required"}}, status_code=411)
                resp.headers["x-request-id"] = rid
                return resp
            if int(cl) > cfg.max_body_bytes:
                resp = JSONResponse({"error": {"message": "request body too large"}}, status_code=413)
                resp.headers["x-request-id"] = rid
                return resp
        resp = await call_next(request)
        resp.headers["x-request-id"] = rid
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Cache-Control"] = "no-store"
        log.info("http", extra={"request_id": rid, "method": request.method, "path": request.url.path,
                                "status": resp.status_code, "ms": round(1000 * (time.perf_counter() - t0), 1)})
        return resp

    @app.get("/health")
    async def health():
        # public liveness probe: no configuration, devices or errors
        return {"status": "ok" if engine.loaded else "loading"}

    @app.get("/v1/info")
    async def info(request: Request):
        _auth(request)
        return {"uptime_s": round(time.time() - state["started"], 1), "queue": state["pending"],
                "version": _version(), **engine.info()}

    @app.get("/metrics")
    async def metrics(request: Request):
        if not cfg.public_metrics:
            _auth(request)
        engine.metrics.set("queue_depth", state["pending"], help="Requests waiting or running")
        return PlainTextResponse(engine.metrics.render(), media_type="text/plain; version=0.0.4")

    @app.get("/v1/models")
    async def models(request: Request):
        _auth(request)
        now = int(state["started"])
        data = [{"id": cfg.model_name, "object": "model", "created": now, "owned_by": "freewidth"}]
        data += [{"id": t, "object": "model", "created": now, "owned_by": "freewidth"} for t in engine.models]
        return {"object": "list", "data": data}

    @app.post("/v1/chat/completions")
    async def chat(req: ChatRequest, request: Request):
        _auth(request)
        rid = request.state.rid
        if len(req.messages) > cfg.max_messages:
            raise HTTPException(status_code=400, detail=f"too many messages (limit {cfg.max_messages})")
        try:
            system, question = split_messages(req.messages)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        if len(question) + len(system or "") > cfg.max_prompt_chars:
            raise HTTPException(status_code=400, detail=f"prompt too long (limit {cfg.max_prompt_chars} characters)")
        overrides = dict(req.freewidth or {})
        bad = set(overrides) - _ALLOWED_OVERRIDES
        if bad:
            raise HTTPException(status_code=400, detail=f"unknown freewidth options: {sorted(bad)}")
        raw = bool(overrides.pop("raw_prompt", not engine.cfg.prompt.apply_to_chat))
        for k, v in (("temperature", req.temperature), ("top_p", req.top_p),
                     ("max_new_tokens", req.max_completion_tokens or req.max_tokens)):
            if v is not None:
                overrides[k] = v
        enforce_limits(cfg, overrides)
        if state["pending"] >= cfg.max_queue:
            engine.metrics.inc("rejected_total", 1, {"reason": "queue_full"}, help="Rejected requests")
            raise HTTPException(status_code=503, detail="server busy: queue full", headers={"Retry-After": "5"})
        state["pending"] += 1
        deadline = time.time() + cfg.request_timeout_s
        try:
            async with gate:
                res = await asyncio.to_thread(engine.solve, question, system=system, raw_prompt=raw,
                                              overrides=overrides, seed=req.seed, deadline=deadline,
                                              request_id=rid)
        except DeadlineExceeded as e:
            engine.metrics.inc("rejected_total", 1, {"reason": "timeout"}, help="Rejected requests")
            raise HTTPException(status_code=504, detail=str(e)) from e
        except ValueError as e:  # bad override values
            raise HTTPException(status_code=400, detail=str(e)) from e
        finally:
            state["pending"] -= 1
        created = int(time.time())
        name = req.model or cfg.model_name
        body = {
            "id": f"chatcmpl-{rid}", "object": "chat.completion", "created": created, "model": name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": res.text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": res.prompt_tokens, "completion_tokens": res.tokens,
                      "total_tokens": res.prompt_tokens + res.tokens},
            "freewidth": res.to_dict(),
        }
        if not req.stream:
            return JSONResponse(body)

        def sse():
            chunk = {"id": body["id"], "object": "chat.completion.chunk", "created": created, "model": name,
                     "choices": [{"index": 0, "delta": {"role": "assistant", "content": res.text},
                                  "finish_reason": None}]}
            yield f"data: {json.dumps(chunk)}\n\n"
            end = {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                   "freewidth": body["freewidth"], "usage": body["usage"]}
            yield f"data: {json.dumps(end)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        log.exception("unhandled error", extra={"request_id": getattr(request.state, "rid", None)})
        return JSONResponse({"error": {"message": "internal error",
                                       "request_id": getattr(request.state, "rid", None)}}, status_code=500)

    return app


def _version() -> str:
    from .. import __version__

    return __version__


def serve(config_path: str, host: str | None = None, port: int | None = None,
          log_json: bool | None = None, log_level: str | None = None,
          insecure_no_auth: bool = False) -> None:
    import uvicorn

    from ..logs import setup_logging

    engine = Engine.from_yaml(config_path, load=False)
    scfg = engine.cfg.server
    if host:
        scfg.host = host
    if insecure_no_auth:
        scfg.allow_unauthenticated = True
    check_exposure(scfg, bool(os.environ.get(scfg.api_key_env)))   # before loading any model
    setup_logging(log_level or engine.cfg.log_level, engine.cfg.log_json or bool(log_json))
    engine.load()
    app = create_app(engine)
    try:
        uvicorn.run(app, host=scfg.host, port=port or scfg.port, log_level="warning",
                    access_log=False, server_header=False)
    finally:
        engine.close()

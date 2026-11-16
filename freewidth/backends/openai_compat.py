"""OpenAI-compatible chat endpoint: vLLM, llama.cpp server, Ollama, Groq, Gemini,
OpenRouter, Cerebras, ...

`measures_device` is True only for servers you run yourself (vLLM / llama.cpp /
Ollama on the same machine); for remote free-tier APIs wall-clock is network
latency and is recorded but not used as device cost.
"""
from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from ..costs import ModelSpec
from .base import Backend, GenResult

_LOOPBACK = {"localhost", "127.0.0.1", "::1"}


def _check_base_url(url: str, sends_key: bool, allow_insecure_http: bool) -> str:
    """Only http(s) URLs (no file:, ftp:, ...). An API key is never sent in clear text to a
    non-loopback host unless the config explicitly allows it."""
    from urllib.parse import urlparse

    u = urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise ValueError(f"base_url must be an http(s) URL with a host, got {url!r}")
    if u.scheme == "http" and sends_key and u.hostname not in _LOOPBACK and not allow_insecure_http:
        raise ValueError(f"refusing to send an API key over plain http to {u.hostname}; use https "
                         "or set allow_insecure_http: true")
    return url.rstrip("/")


class OpenAICompatBackend(Backend):
    def __init__(self, tag: str, model_id: str, base_url: str,
                 api_key_env: str | None = None, supports_n: bool = True,
                 max_concurrency: int = 4, rpm: float | None = None,
                 local: bool = False, spec: dict | None = None,
                 extra_body: dict | None = None, logprobs: bool = False,
                 timeout: float = 600.0, max_retries: int = 6,
                 allow_insecure_http: bool = False, **_):
        self.tag = tag
        self.model_id = model_id
        self.base_url = _check_base_url(base_url, bool(api_key_env), allow_insecure_http)
        self.api_key = os.environ.get(api_key_env, "") if api_key_env else "none"
        self.supports_n = supports_n
        self.max_concurrency = max_concurrency
        self.min_interval = 60.0 / rpm if rpm else 0.0
        self._last_call = 0.0
        self.measures_device = local
        self._spec = ModelSpec.from_dict({"name": tag, **spec}) if spec else None
        self.extra_body = extra_body or {}
        self.logprobs = logprobs
        self.timeout = timeout
        self.max_retries = max_retries

    # ------------------------------------------------------------------ http
    def _post(self, payload: dict) -> dict:
        url = f"{self.base_url}/chat/completions"
        data = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            wait = self._last_call + self.min_interval - time.time()
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.time()
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:  # nosec B310: scheme checked in _check_base_url
                    return json.loads(r.read().decode())
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    ra = e.headers.get("Retry-After") if e.headers else None
                    time.sleep(float(ra) if ra and ra.replace(".", "").isdigit() else delay)
                    delay = min(delay * 2, 120) * (1 + 0.1 * random.random())
                    continue
                body = e.read().decode(errors="replace")[:500]
                raise RuntimeError(f"{self.tag}: HTTP {e.code}: {body}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay = min(delay * 2, 120)
                    continue
                raise RuntimeError(f"{self.tag}: {e}") from e
        raise RuntimeError(f"{self.tag}: retries exhausted")

    def _payload(self, system, user, n, temperature, top_p, max_tokens, seed):
        p = {
            "model": self.model_id,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens,
        }
        if n > 1:
            p["n"] = n
        if seed is not None:
            p["seed"] = seed
        if self.logprobs:
            p["logprobs"] = True
        p.update(self.extra_body)
        return p

    @staticmethod
    def _parse(resp: dict):
        texts, finish, lps, ntoks = [], [], [], []
        for ch in resp.get("choices", []):
            msg = ch.get("message") or {}
            texts.append(msg.get("content") or "")
            finish.append(ch.get("finish_reason") or "")
            lp = (ch.get("logprobs") or {}).get("content") if ch.get("logprobs") else None
            if lp:
                vals = [t.get("logprob", 0.0) for t in lp]
                lps.append(sum(vals) / max(1, len(vals)))
                ntoks.append(len(vals))
            else:
                lps.append(None)
                ntoks.append(None)
        usage = resp.get("usage") or {}
        return texts, finish, lps, ntoks, usage

    # -------------------------------------------------------------- generate
    def generate(self, system, user, n, *, temperature, top_p, max_new_tokens,
                 seed=None, question=None) -> GenResult:
        t0 = time.perf_counter()
        if self.supports_n or n == 1:
            resps = [self._post(self._payload(system, user, n, temperature, top_p, max_new_tokens, seed))]
        else:
            def one(i):
                s = None if seed is None else seed * 1000 + i
                return self._post(self._payload(system, user, 1, temperature, top_p, max_new_tokens, s))

            with ThreadPoolExecutor(max_workers=min(n, self.max_concurrency)) as ex:
                resps = list(ex.map(one, range(n)))
        wall = time.perf_counter() - t0
        texts, finish, lps, ntoks = [], [], [], []
        comp_total, prompt_tokens = 0, 0
        for r in resps:
            t, f, lp, nt, usage = self._parse(r)
            texts += t
            finish += f
            lps += lp
            ntoks += nt
            comp_total += int(usage.get("completion_tokens") or 0)
            prompt_tokens = int(usage.get("prompt_tokens") or prompt_tokens)
        # per-sample token counts: exact from logprobs if present, else split usage by length
        if any(x is None for x in ntoks):
            chars = [max(1, len(t)) for t in texts]
            tot = sum(chars)
            est = comp_total if comp_total else sum(c // 4 for c in chars)
            ntoks = [max(1, round(est * c / tot)) for c in chars]
        return GenResult(texts, ntoks, prompt_tokens, finish, [None] * len(texts),
                         [None] * len(texts), lps, wall, None)

    def score_yes(self, system, users, batch=8, questions=None):
        """P(Yes) from the first output token's top logprobs (vLLM / OpenAI style)."""
        import math as _m

        def one(u):
            p = self._payload(system, u, 1, 0.0, 1.0, 1, 0)
            p.update({"logprobs": True, "top_logprobs": 10})
            r = self._post(p)
            ch = (r.get("choices") or [{}])[0]
            top = (((ch.get("logprobs") or {}).get("content") or [{}])[0]).get("top_logprobs") or []
            py = sum(_m.exp(x["logprob"]) for x in top if x.get("token", "").strip().lower() == "yes")
            pn = sum(_m.exp(x["logprob"]) for x in top if x.get("token", "").strip().lower() == "no")
            ptok = int((r.get("usage") or {}).get("prompt_tokens") or len(u) // 4)
            return (py / (py + pn) if py + pn > 0 else 0.5), ptok

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as ex:
            return list(ex.map(one, users))

    def timed_decode(self, width, gen_len, prompt_len):
        user = " ".join(["token"] * max(1, prompt_len - 20))
        payload = self._payload("Repeat the word 'ok' forever.", user, width, 0.0, 1.0, gen_len, 0)
        payload["ignore_eos"] = True  # vLLM extension; ignored elsewhere
        payload["min_tokens"] = gen_len
        if not self.supports_n and width > 1:
            payload.pop("n", None)
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=width) as ex:
                list(ex.map(lambda _: self._post(payload), range(width)))
            return time.perf_counter() - t0, None
        t0 = time.perf_counter()
        self._post(payload)
        return time.perf_counter() - t0, None

    def spec(self):
        return self._spec

    def device_name(self) -> str:
        return f"remote:{self.base_url}" if not self.measures_device else f"local-server:{self.base_url}"

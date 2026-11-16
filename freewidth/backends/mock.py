"""Deterministic simulated model for tests and offline smoke runs.

Accuracy depends on model size and question difficulty; wrong answers come from
a small pool of distractors so that voting behaves realistically. Wall-clock is
synthesised from a roofline-like curve so that the profiler and replay code
can be exercised end to end without a GPU.
"""
from __future__ import annotations

import hashlib
import math
import random

from ..costs import ModelSpec
from .base import Backend, GenResult


def _h(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


class MockBackend(Backend):
    measures_device = True

    def __init__(self, tag: str, params_b: float = 1.0, skill: float | None = None,
                 n_layers: int = 24, hybrid_ratio: float = 0.0, **_):
        self.tag = tag
        self.params = params_b * 1e9
        # bigger models are better, with diminishing returns
        self.skill = skill if skill is not None else 0.15 + 0.14 * math.log2(1 + params_b)
        self.n_layers = n_layers
        self.hybrid_ratio = hybrid_ratio
        # synthetic device: weight-bandwidth bound step cost + per-row compute
        self._a = 2.0e-3 * params_b          # per step, shared by the wave
        self._b = 0.06e-3 * params_b         # per step per row
        self._g = 1.0e-8 * (1 - hybrid_ratio) * n_layers / 24  # per ctx token
        self._d = 0.02e-3 * params_b
        self._e = 0.05
        self._watts = 70.0

    def _p_correct(self, q) -> float:
        level = (q.meta or {}).get("level", 3) if q is not None else 3
        try:
            level = float(level if not isinstance(level, str) else level.split()[-1])
        except (TypeError, ValueError):
            level = 3.0
        noise = (_h(self.tag, q.qid if q else "") % 1000) / 1000.0 - 0.5
        z = 4.0 * (self.skill - 0.12 * level) + 1.5 * noise
        return 1.0 / (1.0 + math.exp(-z))

    def _secs(self, w, L, P):
        return (self._e + self._d * w * P + self._a * L + self._b * w * L
                + self._g * w * (L * P + L * (L + 1) / 2))

    def generate(self, system, user, n, *, temperature, top_p, max_new_tokens,
                 seed=None, question=None) -> GenResult:
        rng = random.Random(_h(self.tag, question.qid if question else user, seed))
        p = self._p_correct(question)
        gold = question.answer if question else "0"
        distractors = [str(int(gold) + d) if gold.lstrip("-").isdigit() else f"{gold}_{d}"
                       for d in (1, -1, 2, 10)]
        texts, lens, sc, ent = [], [], [], []
        prompt_tokens = max(8, len(user) // 4)
        for _ in range(n):
            ok = rng.random() < p
            ans = gold if ok else rng.choice(distractors[: 2 + int(3 * (1 - p))])
            L = min(max_new_tokens, int(rng.gauss(180, 50) * (1.3 if not ok else 1.0)) + 20)
            L = max(L, 16)
            texts.append(f"... reasoning ... so the result is \\boxed{{{ans}}}")
            lens.append(L)
            # confident when correct, a bit less when wrong
            sc.append(rng.gauss(1.9 if ok else 1.7, 0.4))
            ent.append(rng.gauss(0.6 if ok else 0.8, 0.1))
        wall = self._secs(n, max(lens), prompt_tokens)
        return GenResult(texts, lens, prompt_tokens, ["stop"] * n, sc, ent, [None] * n,
                         wall, wall * self._watts)

    def score_yes(self, system, users, batch=8, questions=None):
        """Simulated verifier: separates right from wrong with skill-dependent noise."""
        from ..grading import extract_answer, is_equiv

        out = []
        for u, q in zip(users, questions or [None] * len(users)):
            rng = random.Random(_h(self.tag, "verify", u))
            ok = q is not None and is_equiv(extract_answer(u.split("Proposed solution:")[-1]), q.answer)
            z = (1.2 + 1.5 * self.skill) * (1 if ok else -1) + rng.gauss(0, 1.3)
            out.append((1 / (1 + math.exp(-z)), max(8, len(u) // 4)))
        return out

    def timed_decode(self, width, gen_len, prompt_len):
        rng = random.Random(_h(self.tag, width, gen_len, prompt_len))
        s = self._secs(width, gen_len, prompt_len) * (1 + rng.gauss(0, 0.02))
        return s, s * self._watts

    def spec(self) -> ModelSpec:
        full = int(round(self.n_layers * (1 - self.hybrid_ratio)))
        return ModelSpec(self.tag, self.params, n_layers=self.n_layers,
                         n_full_attn_layers=full, n_heads=16, n_kv_heads=4, head_dim=128,
                         linear_state_bytes=(self.n_layers - full) * 16 * 128 * 128 * 4)

    def device_name(self) -> str:
        return "mock-device"

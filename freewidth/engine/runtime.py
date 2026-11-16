"""Live wave generation with an online QuorumStop.

One call = one lock-step wave of `n` parallel samples on one device. Rows finish at
different times; after every decode step the stopping criterion looks at rows that
just emitted EOS, extracts their final answers and, once the finished rows (plus any
votes carried over from earlier waves) form a confident quorum, stops the whole wave.
Rows still running are cut: they cost what they consumed and do not vote.
"""
from __future__ import annotations

import math
import threading
import time
from collections import Counter
from dataclasses import dataclass, field

from ..grading import answer_key, extract_answer
from ..strategies import lead_probability


@dataclass
class Row:
    text: str
    key: str | None
    gen_tokens: int
    finish: str                    # stop | length | quorum_cut
    self_certainty: float | None = None
    entropy: float | None = None

    @property
    def votes(self) -> bool:
        return self.finish != "quorum_cut" and self.key is not None


@dataclass
class WaveResult:
    rows: list[Row]
    prompt_tokens: int
    wall_s: float
    energy_j: float | None = None
    quorum_hit: bool = False
    stop_step: int | None = None
    model: str = ""

    @property
    def voting_rows(self) -> list[Row]:
        return [r for r in self.rows if r.votes]

    @property
    def tokens(self) -> int:
        return sum(r.gen_tokens for r in self.rows)


@dataclass
class QuorumState:
    tau: float
    prior: Counter = field(default_factory=Counter)
    min_finished: int = 2

    def confident(self, counts: Counter) -> bool:
        total = counts + self.prior
        return sum(total.values()) >= self.min_finished and lead_probability(total) >= self.tau


class SharedQuorum:
    """One QuorumStop shared by concurrent waves on different devices: votes from every wave are
    pooled, and all waves stop as soon as the pooled votes are confident."""

    def __init__(self, state: QuorumState):
        self.state = state
        self.counts: Counter = Counter()
        self.hit = False
        self._lock = threading.Lock()

    def add(self, keys) -> bool:
        with self._lock:
            for k in keys:
                if k is not None:
                    self.counts[k] += 1
            if not self.hit and self.state.confident(self.counts):
                self.hit = True
            return self.hit


class _QuorumCriteria:
    """transformers StoppingCriteria: returns all-True once a quorum forms."""

    def __init__(self, tok, prompt_len: int, eos_ids: set[int], state: QuorumState | None,
                 shared: SharedQuorum | None = None):
        self.tok = tok
        self.P = prompt_len
        self.eos = eos_ids
        self.state = state
        self.shared = shared
        self.finished: dict[int, str | None] = {}
        self.hit = False
        self.step = 0

    def __call__(self, input_ids, scores, **kwargs):
        import torch

        self.step += 1
        B = input_ids.shape[0]
        last = input_ids[:, -1].tolist()
        new = [i for i in range(B) if i not in self.finished and last[i] in self.eos]
        for i in new:
            text = self.tok.decode(input_ids[i, self.P:], skip_special_tokens=True)
            self.finished[i] = answer_key(extract_answer(text))
        if self.shared is not None:
            self.hit = self.shared.add([self.finished[i] for i in new]) if new else self.shared.hit
        elif new and not self.hit:
            counts = Counter(k for k in self.finished.values() if k is not None)
            if self.state.confident(counts):
                self.hit = True
        return torch.full((B,), self.hit, dtype=torch.bool, device=input_ids.device)


def hf_generate_wave(backend, system: str, user: str, n: int, *, temperature: float,
                     top_p: float, max_new_tokens: int, seed: int | None = None,
                     quorum: QuorumState | None = None,
                     shared: SharedQuorum | None = None) -> WaveResult:
    """Run one wave on an `HFBackend`, optionally with a live QuorumStop (own or shared)."""
    from transformers import StoppingCriteriaList

    from ..backends.hf import _StatsRecorder

    t = backend._torch
    enc = backend._encode(system, user)
    P = int(enc["input_ids"].shape[1])
    eos = backend._eos_ids()
    rec = _StatsRecorder()
    crit = _QuorumCriteria(backend.tok, P, eos, quorum, shared) if (quorum or shared) else None
    if seed is not None:
        backend._seed(seed)
    backend._sync()
    backend.meter.start()
    t0 = time.perf_counter()
    with t.inference_mode():
        out = backend.model.generate(
            **enc, do_sample=temperature > 0, temperature=max(temperature, 1e-5), top_p=top_p,
            top_k=0, max_new_tokens=max_new_tokens, num_return_sequences=n,
            logits_processor=[rec], pad_token_id=backend.tok.pad_token_id,
            stopping_criteria=StoppingCriteriaList([crit]) if crit else None)
    backend._sync()
    wall = time.perf_counter() - t0
    joules = backend.meter.stop()
    gen = out[:, P:].cpu().tolist()
    hit = bool(crit and crit.hit)
    ent = t.stack(rec.ent, dim=1) if rec.ent else None
    sc = t.stack(rec.sc, dim=1) if rec.sc else None
    rows = []
    for i, ids in enumerate(gen):
        L = next((k + 1 for k, tok in enumerate(ids) if tok in eos), None)
        if L is None:
            L = len(ids)
            finish = "quorum_cut" if hit else "length"
        else:
            finish = "stop"
        text = backend.tok.decode(ids[:L], skip_special_tokens=True)
        key = answer_key(extract_answer(text)) if finish != "quorum_cut" else None
        e = float(ent[i, :L].mean()) if ent is not None and L else None
        s = float(sc[i, :L].mean()) if sc is not None and L else None
        rows.append(Row(text, key, L, finish, s if s is None or math.isfinite(s) else None, e))
    return WaveResult(rows, P, wall, joules, hit, crit.step if hit else None, backend.tag)


def mock_generate_wave(backend, system: str, user: str, n: int, *, temperature: float,
                       top_p: float, max_new_tokens: int, seed: int | None = None,
                       quorum: QuorumState | None = None, question=None,
                       shared: SharedQuorum | None = None) -> WaveResult:
    """Same contract as `hf_generate_wave`, on the simulated backend. A shared quorum is
    approximated by this wave's own votes plus whatever the other waves already pooled."""
    if shared is not None and quorum is None:
        quorum = QuorumState(shared.state.tau, Counter(shared.counts), shared.state.min_finished)
    r = backend.generate(system, user, n, temperature=temperature, top_p=top_p,
                         max_new_tokens=max_new_tokens, seed=seed, question=question)
    rows = [Row(tx, answer_key(extract_answer(tx)), L, "stop", sc, en)
            for tx, L, sc, en in zip(r.texts, r.gen_tokens, r.self_certainty, r.entropy)]
    hit, cut = False, None
    if quorum:
        counts: Counter = Counter()
        for i in sorted(range(n), key=lambda i: (rows[i].gen_tokens, i)):
            if rows[i].key is not None:
                counts[rows[i].key] += 1
            if quorum.confident(counts):
                hit, cut = True, rows[i].gen_tokens
                break
    if hit:
        for r_ in rows:
            if r_.gen_tokens > cut:
                r_.gen_tokens, r_.finish, r_.key = cut, "quorum_cut", None
    if shared is not None:
        shared.add([r_.key for r_ in rows if r_.finish == "stop"])
    L = max(r_.gen_tokens for r_ in rows)
    wall = backend._secs(n, L, r.prompt_tokens)
    return WaveResult(rows, r.prompt_tokens, wall, wall * backend._watts, hit, cut, backend.tag)


def generate_wave(backend, *args, **kwargs) -> WaveResult:
    from ..backends.hf import HFBackend
    from ..backends.mock import MockBackend

    if isinstance(backend, HFBackend):
        kwargs.pop("question", None)
        return hf_generate_wave(backend, *args, **kwargs)
    if isinstance(backend, MockBackend):
        return mock_generate_wave(backend, *args, **kwargs)
    raise TypeError(f"live waves are not supported for {type(backend).__name__}")

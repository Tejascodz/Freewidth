"""Overlap: a two-device ladder that uses the idle GPU while the small model decodes.

Policy `overlap` (proposed). Small model S on device 0, big model B on device 1.
While S decodes one free-width wave, B is otherwise idle; Overlap gives it two jobs:

1. Streaming verification. Every S row that finishes is sent at once to B, which
   prefill-scores P(Yes | "is this final answer correct?"). Verification runs while S is
   still decoding, so it adds no wall time unless the decision waits for it.
2. Speculative escalation. When the wave's own dynamics predict that S will fail, B starts
   decoding its own wave immediately, in parallel, instead of after S gives up. The
   predictors come from the T4 runs: escalated questions had 0/18 first-wave quorums and
   ~6 of 16 rows still running at the length cap, against 0.3 otherwise. The triggers are:
   finished rows disagree, the leading answer is vetoed by B, or stragglers outlive the
   finished rows by `runaway_factor`.

Decision rule on S's finished rows, after every decode step:
  accept lead     if posterior(lead) >= tau and B has not vetoed it
                  (and, with require_verified, B has scored at least one of its chains)
  accept verified if some answer has >= 2 votes and B's best score for it >= accept_hi
  veto            if B scored every chain of the lead below veto_lo -> no accept, escalate
  cut stragglers  once >= runaway_min_finished rows are done, rows still running beyond
                  runaway_factor x the median finished length stop and do not vote

If S accepts first, B's speculative wave is cancelled. Otherwise B's waves decide, ties
broken by B's verification scores of S's candidates.

The controller is pure Python and step-driven, so the same logic runs inside a
transformers StoppingCriteria (live) and in an event simulation (mock backend, tests).
"""
from __future__ import annotations

import statistics
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from ..grading import answer_key, extract_answer
from ..logs import get_logger
from ..strategies import lead_probability

log = get_logger("overlap")


@dataclass
class OverlapDecision:
    accept: str | None = None
    reason: str = ""
    cut_rows: set = field(default_factory=set)
    trigger: bool = False
    trigger_reason: str = ""


class OverlapController:
    """Step-driven decision logic for one small-model wave."""

    def __init__(self, width: int, tau: float, *, veto_lo: float = 0.2, accept_hi: float = 0.95,
                 require_verified: bool = True, min_finished: int = 2, trigger_min_finished: int = 4,
                 trigger_lead_prob: float = 0.6, runaway_factor: float = 2.0,
                 runaway_min_finished: int = 4, speculative: bool = True):
        self.width = width
        self.tau = tau
        self.veto_lo = veto_lo
        self.accept_hi = accept_hi
        self.require_verified = require_verified
        self.min_finished = min_finished
        self.trigger_min_finished = trigger_min_finished
        self.trigger_lead_prob = trigger_lead_prob
        self.runaway_factor = runaway_factor
        self.runaway_min_finished = runaway_min_finished
        self.speculative = speculative
        self.counts: Counter = Counter()
        self.finished: dict = {}          # row -> (key, length, step)
        self.texts: dict = {}             # row -> text
        self.pyes: dict = defaultdict(list)   # key -> [p_yes of verified chains]
        self.cut: set = set()
        self.triggered = False
        self.trigger_step: int | None = None
        self.trigger_reason = ""

    # ------------------------------------------------------------- events
    def on_finish(self, row: int, text: str, length: int, step: int) -> str | None:
        key = answer_key(extract_answer(text))
        self.finished[row] = (key, length, step)
        self.texts[row] = text
        if key is not None:
            self.counts[key] += 1
        return key

    def on_verified(self, key: str | None, p: float) -> None:
        if key is not None:
            self.pyes[key].append(float(p))

    # ------------------------------------------------------------ queries
    def best_p(self, key) -> float | None:
        ps = self.pyes.get(key)
        return max(ps) if ps else None

    def vetoed(self, key) -> bool:
        p = self.best_p(key)
        return p is not None and p < self.veto_lo

    def lead(self) -> tuple[str | None, float]:
        if not self.counts:
            return None, 0.0
        return self.counts.most_common(1)[0][0], lead_probability(self.counts)

    def running_rows(self) -> list[int]:
        return [r for r in range(self.width) if r not in self.finished and r not in self.cut]

    # ------------------------------------------------------------ decide
    def decide(self, step: int) -> OverlapDecision:
        d = OverlapDecision()
        key, lp = self.lead()
        n_fin = len(self.finished)
        # 1. accept by a B-verified answer with support
        verified = [(self.best_p(k), c, k) for k, c in self.counts.items()
                    if c >= 2 and self.best_p(k) is not None and self.best_p(k) >= self.accept_hi]
        if verified:
            d.accept, d.reason = max(verified)[2], "verified"
            return d
        # 2. accept by quorum unless vetoed (optionally after B has looked at it)
        quorum = key is not None and sum(self.counts.values()) >= self.min_finished and lp >= self.tau
        if quorum and not self.vetoed(key) and (not self.require_verified or self.best_p(key) is not None):
            d.accept, d.reason = key, "quorum"
            return d
        # 3. straggler cut: rows outliving the finished ones by runaway_factor
        lengths = [v[1] for v in self.finished.values()]
        if n_fin >= self.runaway_min_finished:
            med = statistics.median(lengths)
            if step > self.runaway_factor * med:
                d.cut_rows = set(self.running_rows())
                self.cut |= d.cut_rows
        # 4. speculative escalation triggers
        if self.speculative and not self.triggered:
            reason = ""
            if key is not None and self.vetoed(key) and self.counts[key] >= 2:
                reason = "lead_vetoed"
            elif n_fin >= self.trigger_min_finished and lp < self.trigger_lead_prob:
                reason = "disagreement"
            elif n_fin >= self.runaway_min_finished and len(self.running_rows()) >= max(2, self.width // 3) \
                    and step > self.runaway_factor * statistics.median(lengths):
                reason = "stragglers"
            if reason:
                self.triggered, self.trigger_step, self.trigger_reason = True, step, reason
                d.trigger, d.trigger_reason = True, reason
        return d

    def end_of_wave(self) -> OverlapDecision:
        """No accept during the wave: final look with everything verified so far."""
        d = OverlapDecision()
        key, lp = self.lead()
        verified = [(self.best_p(k), c, k) for k, c in self.counts.items()
                    if self.best_p(k) is not None and self.best_p(k) >= self.accept_hi]
        if verified:
            d.accept, d.reason = max(verified)[2], "verified_end"
        elif key is not None and lp >= self.tau and not self.vetoed(key):
            d.accept, d.reason = key, "quorum_end"
        return d


# ---------------------------------------------------------------------------
# live (transformers) path
# ---------------------------------------------------------------------------

class _ControllerCriteria:
    """Feeds finished rows to the controller and the verifier; applies its decisions."""

    def __init__(self, tok, prompt_len, eos_ids, ctrl: OverlapController, on_finish, on_trigger, poll):
        self.tok, self.P, self.eos, self.ctrl = tok, prompt_len, eos_ids, ctrl
        self.on_finish, self.on_trigger, self.poll = on_finish, on_trigger, poll
        self.step = 0
        self.accepted: OverlapDecision | None = None

    def __call__(self, input_ids, scores, **kwargs):
        import torch

        self.step += 1
        B = input_ids.shape[0]
        last = input_ids[:, -1].tolist()
        for i in range(B):
            if i not in self.ctrl.finished and i not in self.ctrl.cut and last[i] in self.eos:
                text = self.tok.decode(input_ids[i, self.P:], skip_special_tokens=True)
                key = self.ctrl.on_finish(i, text, self.step, self.step)
                self.on_finish(i, key, text)
        self.poll()  # pull verification results into the controller
        if self.accepted is None:
            d = self.ctrl.decide(self.step)
            if d.trigger:
                self.on_trigger(d.trigger_reason)
            if d.accept is not None:
                self.accepted = d
        if self.accepted is not None:
            return torch.ones(B, dtype=torch.bool, device=input_ids.device)
        stop = torch.zeros(B, dtype=torch.bool, device=input_ids.device)
        for r in self.ctrl.cut:
            stop[r] = True
        return stop


class _Cancel:
    def __init__(self, event: threading.Event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs):
        import torch

        return torch.full((input_ids.shape[0],), self.event.is_set(), dtype=torch.bool,
                          device=input_ids.device)


def hf_wave_with_controller(backend, system, user, n, *, temperature, top_p, max_new_tokens, seed,
                            ctrl: OverlapController, on_finish, on_trigger, poll):
    """One small-model wave under the Overlap controller. Returns (rows, wall_s, joules, decision)."""
    from transformers import StoppingCriteriaList

    from ..backends.hf import _StatsRecorder
    from .runtime import Row

    t = backend._torch
    enc = backend._encode(system, user)
    P = int(enc["input_ids"].shape[1])
    eos = backend._eos_ids()
    crit = _ControllerCriteria(backend.tok, P, eos, ctrl, on_finish, on_trigger, poll)
    if seed is not None:
        backend._seed(seed)
    backend._sync()
    backend.meter.start()
    t0 = time.perf_counter()
    with t.inference_mode():
        out = backend.model.generate(
            **enc, do_sample=temperature > 0, temperature=max(temperature, 1e-5), top_p=top_p, top_k=0,
            max_new_tokens=max_new_tokens, num_return_sequences=n, logits_processor=[_StatsRecorder()],
            pad_token_id=backend.tok.pad_token_id, stopping_criteria=StoppingCriteriaList([crit]))
    backend._sync()
    wall = time.perf_counter() - t0
    joules = backend.meter.stop()
    gen = out[:, P:].cpu().tolist()
    rows = []
    for i, ids in enumerate(gen):
        L = next((k + 1 for k, tok in enumerate(ids) if tok in eos), None)
        if i in ctrl.finished:
            key, L, _ = ctrl.finished[i]
            rows.append(Row(ctrl.texts[i], key, L, "stop"))
        else:
            L = L or len(ids)
            fin = "quorum_cut" if (crit.accepted is not None or i in ctrl.cut) else "length"
            text = backend.tok.decode(ids[:L], skip_special_tokens=True)
            rows.append(Row(text, None if fin == "quorum_cut" else answer_key(extract_answer(text)), L, fin))
    return rows, P, wall, joules, crit.accepted


def hf_cancellable_wave(backend, system, user, n, *, temperature, top_p, max_new_tokens, seed,
                        quorum, cancel: threading.Event):
    """A big-model wave with live QuorumStop that can be cancelled from another thread."""
    from transformers import StoppingCriteriaList

    from ..backends.hf import _StatsRecorder
    from .runtime import Row, WaveResult, _QuorumCriteria

    t = backend._torch
    enc = backend._encode(system, user)
    P = int(enc["input_ids"].shape[1])
    eos = backend._eos_ids()
    q = _QuorumCriteria(backend.tok, P, eos, quorum)
    if seed is not None:
        backend._seed(seed)
    backend._sync()
    backend.meter.start()
    t0 = time.perf_counter()
    with t.inference_mode():
        out = backend.model.generate(
            **enc, do_sample=temperature > 0, temperature=max(temperature, 1e-5), top_p=top_p, top_k=0,
            max_new_tokens=max_new_tokens, num_return_sequences=n, logits_processor=[_StatsRecorder()],
            pad_token_id=backend.tok.pad_token_id,
            stopping_criteria=StoppingCriteriaList([q, _Cancel(cancel)]))
    backend._sync()
    wall = time.perf_counter() - t0
    joules = backend.meter.stop()
    rows = []
    for ids in out[:, P:].cpu().tolist():
        L = next((k + 1 for k, tok in enumerate(ids) if tok in eos), None)
        if L is None:
            L, fin = len(ids), ("quorum_cut" if (q.hit or cancel.is_set()) else "length")
        else:
            fin = "stop"
        text = backend.tok.decode(ids[:L], skip_special_tokens=True)
        rows.append(Row(text, answer_key(extract_answer(text)) if fin == "stop" else None, L, fin))
    return WaveResult(rows, P, wall, joules, q.hit, None, backend.tag)

"""Test-time scaling strategies, all expressed as replays over the sample bank.

Baselines
  single        one sample                                          (CoT)
  sc            n parallel samples + majority vote                  (Wang et al. 2023)
  sc_seq        n samples drawn one at a time + vote                (same FLOPs, n rounds)
  wvote         n samples, Borda vote weighted by self-certainty    (Kang et al. 2025)
  adaptive      one sample at a time, Beta stopping rule            (Aggarwal et al. 2023)
  cascade       vote on small model, escalate if agreement < tau    (FrugalGPT-style)
  proxy_alloc   dataset-level: probe wave, then split a global
                budget by first-wave disagreement                   (DIPA-style proxy)
  short_mk      k parallel, stop all rows when the first m finish,
                vote among those m                                  (Hassid et al. 2025)
  conf_filter   keep the most confident fraction, confidence-weighted
                vote                                                (DeepConf offline, Fu et al. 2026;
                                                                     mean-confidence variant)

Proposed
  hwave         Hardware-aware wave controller: sample in waves sized to the device's
                *free width* (widest batch costing <= (1+eps) x one sample), stop on a
                Beta posterior, escalate to the next model size only after waves fail.
  hwave(quorum) QuorumStop: inside a lock-step wave, rows finish at different times;
                stop the whole wave as soon as the rows finished so far form a
                confident quorum (posterior-adaptive m, unlike the fixed m of short-m@k).
                On static-batch hardware this cuts the straggler tail that dominates
                wave latency.
  pverify       Prefill-Verified Escalation: the big model never decodes. It reads the
                small model's distinct candidate solutions in one batched forward pass
                and scores P("Yes" | is this correct?). Prefill is compute-bound, so on a
                memory-bound device this costs a few decode steps, not a full answer.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import replace

from .replay import Bank, Sample, Stream, Trace


# ------------------------------------------------------------------ voting
def vote(samples: list[Sample], weights: list[float] | None = None) -> tuple[str | None, Counter]:
    c: Counter = Counter()
    first: dict = {}
    for i, s in enumerate(samples):
        if s.key is None:
            continue
        c[s.key] += weights[i] if weights else 1.0
        first.setdefault(s.key, i)
    if not c:
        return None, c
    best = max(c.items(), key=lambda kv: (kv[1], -first[kv[0]]))[0]
    return best, c


def borda_weights(samples: list[Sample], power: float = 1.0) -> list[float] | None:
    vals = [s.sc for s in samples]
    if any(v is None for v in vals):
        return None
    order = sorted(range(len(vals)), key=lambda i: -vals[i])
    w = [0.0] * len(vals)
    n = len(vals)
    for rank, i in enumerate(order):
        w[i] = (n - rank) ** power
    return w


def lead_probability(counts: Counter) -> float:
    """P(leading answer has the highest underlying probability) under a uniform
    Dirichlet prior, using the top-two Beta reduction of Adaptive-Consistency."""
    if not counts:
        return 0.0
    top = counts.most_common(2)
    a = top[0][1] + 1.0
    b = (top[1][1] if len(top) > 1 else 0.0) + 1.0
    try:
        from scipy.stats import beta

        return float(beta.sf(0.5, a, b))
    except Exception:  # pragma: no cover
        # normal approximation of the Beta tail
        m = a / (a + b)
        v = a * b / ((a + b) ** 2 * (a + b + 1))
        return 0.5 * (1 + math.erf((m - 0.5) / math.sqrt(2 * v)))


def quorum_wave(samples: list[Sample], tau: float, prior: list[Sample] | None = None,
                min_finished: int = 2):
    """Replay a lock-step wave in finishing order and stop at the first confident quorum.

    Returns (key, priced_samples, stopped). Rows still running at the stop are truncated to
    the stop length and do not vote."""
    prior = prior or []
    order = sorted(range(len(samples)), key=lambda i: (samples[i].gen, i))
    cut = None
    finished: list[Sample] = []
    for i in order:
        finished.append(samples[i])
        if len(prior) + len(finished) >= min_finished:
            _, c = vote(prior + finished)
            if lead_probability(c) >= tau:
                cut = samples[i].gen
                break
    if cut is None:
        key, _ = vote(prior + samples)
        return key, samples, False
    done = [s for s in samples if s.gen <= cut]
    priced = [s if s.gen <= cut else replace(s, gen=cut, key=None) for s in samples]
    key, _ = vote(prior + done)
    return key, priced, True


# --------------------------------------------------------------- strategies
class Strategy:
    family = "base"

    def __init__(self, **params):
        self.params = params

    @property
    def name(self) -> str:
        def fmt(v):
            return "+".join(map(str, v)) if isinstance(v, (list, tuple)) else str(v)

        inner = ",".join(f"{k}={fmt(v)}" for k, v in self.params.items())
        return f"{self.family}({inner})"

    def models(self) -> list[str]:
        m = self.params.get("model") or self.params.get("models") or []
        return [m] if isinstance(m, str) else list(m)

    def run(self, bank: Bank, stream: Stream, qid: str) -> Trace:
        raise NotImplementedError

    def run_dataset(self, bank: Bank, stream: Stream, qids: list[str]) -> dict:
        return {q: self.run(bank, stream, q) for q in qids}


class Single(Strategy):
    family = "single"

    def run(self, bank, stream, qid):
        s = stream.take(self.params["model"], qid, 1)
        tr = Trace(s[0].key if s else None)
        tr.add(self.params["model"], s)
        return tr


class SC(Strategy):
    family = "sc"

    def run(self, bank, stream, qid):
        m, n = self.params["model"], self.params["n"]
        s = stream.take(m, qid, n)
        tr = Trace(vote(s)[0])
        tr.add(m, s)
        return tr


class SCSeq(Strategy):
    """Same samples as SC but issued one per round (e.g. sequential refinement budgets,
    or a server without parallel sampling). Identical FLOPs, very different device cost."""

    family = "sc_seq"

    def run(self, bank, stream, qid):
        m, n = self.params["model"], self.params["n"]
        tr = Trace(None)
        allv = []
        for _ in range(n):
            s = stream.take(m, qid, 1)
            if not s:
                break
            tr.add(m, s)
            allv += s
        tr.key = vote(allv)[0]
        return tr


class WVote(Strategy):
    family = "wvote"

    def run(self, bank, stream, qid):
        m, n = self.params["model"], self.params["n"]
        s = stream.take(m, qid, n)
        tr = Trace(vote(s, borda_weights(s, self.params.get("power", 1.0)))[0])
        tr.add(m, s)
        return tr


class Adaptive(Strategy):
    family = "adaptive"

    def run(self, bank, stream, qid):
        m, tau, nmax = self.params["model"], self.params["tau"], self.params["max_n"]
        tr = Trace(None)
        allv: list[Sample] = []
        while len(allv) < nmax:
            s = stream.take(m, qid, 1)
            if not s:
                break
            tr.add(m, s)
            allv += s
            key, c = vote(allv)
            if len(allv) >= 2 and lead_probability(c) >= tau:
                break
        tr.key = vote(allv)[0]
        return tr


class Cascade(Strategy):
    family = "cascade"

    def run(self, bank, stream, qid):
        small, big = self.params["models"]
        n1, n2, tau = self.params["n_small"], self.params["n_big"], self.params["tau"]
        s1 = stream.take(small, qid, n1)
        tr = Trace(None)
        tr.add(small, s1)
        key, c = vote(s1)
        agree = (c[key] / len(s1)) if (key is not None and s1) else 0.0
        if agree >= tau:
            tr.key = key
            return tr
        s2 = stream.take(big, qid, n2)
        tr.add(big, s2)
        tr.key = vote(s2)[0] if s2 else key
        return tr


class HWave(Strategy):
    """Hardware-aware wave controller (proposed).

    For each model in ascending size:
        w = device free width (or fixed `width`)
        repeat up to `max_waves`: draw a wave of w samples, update votes,
            stop and answer if lead_probability >= tau
    If no model reaches tau, answer with the pooled vote of the largest model tried,
    tie-broken by the smaller models' votes.
    """

    family = "hwave"

    def _width(self, bank, tag) -> int:
        w = self.params.get("width", "free")
        if w == "free":
            fw = bank.models[tag].free_width
            return int(fw) if fw else int(self.params.get("fallback_width", 8))
        return int(w)

    def run(self, bank, stream, qid):
        models = self.models()
        tau = self.params["tau"]
        max_waves = self.params.get("max_waves", 2)
        weighted = self.params.get("weighted", False)
        tr = Trace(None)
        pooled: dict = {}
        for tag in models:
            w = self._width(bank, tag)
            got: list[Sample] = []
            for _ in range(max_waves):
                s = stream.take(tag, qid, w)
                if not s:
                    break
                if self.params.get("quorum"):
                    key, priced, stopped = quorum_wave(s, tau, got)
                    tr.add(tag, priced)
                    if stopped:
                        tr.key = key
                        return tr
                    got += s
                    continue
                tr.add(tag, s)
                got += s
                key, c = vote(got, borda_weights(got) if weighted else None)
                if lead_probability(c if not weighted else vote(got)[1]) >= tau:
                    tr.key = key
                    return tr
            pooled[tag] = got
        last = [t for t in models if pooled.get(t)]
        if not last:
            return tr
        big = pooled[last[-1]]
        key, c = vote(big)
        if len(last) > 1 and len(c) > 1:
            top = c.most_common(2)
            if top[0][1] == top[1][1]:  # tie on the big model: let smaller models break it
                small_votes = vote([s for t in last[:-1] for s in pooled[t]])[1]
                key = max((top[0][0], top[1][0]), key=lambda k: small_votes.get(k, 0))
        tr.key = key
        return tr


class ShortMK(Strategy):
    family = "short_mk"

    def run(self, bank, stream, qid):
        m_tag, k, m = self.params["model"], self.params["k"], self.params["m"]
        s = stream.take(m_tag, qid, k)
        tr = Trace(None)
        if not s:
            return tr
        order = sorted(range(len(s)), key=lambda i: (s[i].gen, i))
        first = [s[i] for i in order[:m]]
        cut = first[-1].gen
        tr.add(m_tag, [x if x.gen <= cut else replace(x, gen=cut, key=None) for x in s])
        tr.key = vote(first)[0]   # list is length-ordered, so ties go to the shortest chain
        return tr


class ConfFilter(Strategy):
    family = "conf_filter"

    def run(self, bank, stream, qid):
        m, n, keep = self.params["model"], self.params["n"], self.params["keep"]
        s = stream.take(m, qid, n)
        tr = Trace(None)
        tr.add(m, s)
        if any(x.sc is None for x in s):
            tr.key = vote(s)[0]
            return tr
        kept = sorted(s, key=lambda x: -x.sc)[: max(1, int(round(keep * len(s))))]
        tr.key = vote(kept, [max(1e-6, x.sc) for x in kept])[0]
        return tr


class PVerify(Strategy):
    """Prefill-Verified Escalation (proposed)."""

    family = "pverify"

    def run(self, bank, stream, qid):
        small, big = self.params["models"]
        n = self.params.get("n", "free")
        if n == "free":
            n = int(bank.models[small].free_width or 8)
        gate = self.params.get("gate_tau")
        mode = self.params.get("mode", "mix")
        top_c = self.params.get("top_c", 4)
        s = stream.take(small, qid, n)
        tr = Trace(None)
        tr.add(small, s)
        key, c = vote(s)
        tr.key = key
        if key is None or len(c) == 1 or (gate is not None and lead_probability(c) >= gate):
            return tr                       # unanimous or confident: no verification pass
        scores = bank.models[big].verify.get(small, {}).get(qid, {})
        cands = [k for k, _ in c.most_common(top_c)]
        reps = {k: next(x for x in s if x.key == k) for k in cands}
        got = {k: scores.get(reps[k].sid) for k in cands}
        got = {k: v for k, v in got.items() if v is not None}
        if not got:
            return tr
        pv = int(sum(v[1] for v in got.values()) / len(got))
        tr.add_prefill(big, len(got), pv)
        total = sum(c.values())
        if mode == "max":
            tr.key = max(got, key=lambda k: (got[k][0], c[k]))
        else:  # mix: vote share + verifier probability
            tr.key = max(got, key=lambda k: c[k] / total + got[k][0])
        return tr


class ProxyAlloc(Strategy):
    """Global budget of `avg_n` samples per question: probe every question with
    `probe` samples, then hand out the rest in proportion to first-wave
    disagreement (questions where all probes agree get nothing more)."""

    family = "proxy_alloc"

    def run_dataset(self, bank, stream, qids):
        m, avg_n, probe = self.params["model"], self.params["avg_n"], self.params.get("probe", 2)
        cap = self.params.get("cap", 32)
        traces, probes, dis = {}, {}, {}
        for q in qids:
            s = stream.take(m, q, probe)
            tr = Trace(None)
            tr.add(m, s)
            traces[q], probes[q] = tr, s
            key, c = vote(s)
            dis[q] = 1.0 - (c[key] / len(s) if key is not None and s else 0.0)
        extra_total = max(0, int(round((avg_n - probe) * len(qids))))
        tot = sum(dis.values())
        alloc = defaultdict(int)
        if tot > 0:
            for q in qids:
                alloc[q] = min(cap - probe, int(round(extra_total * dis[q] / tot)))
        for q in qids:
            s = stream.take(m, q, alloc[q]) if alloc[q] > 0 else []
            traces[q].add(m, s)
            traces[q].key = vote(probes[q] + s)[0]
        return traces

    def run(self, bank, stream, qid):  # pragma: no cover - dataset level only
        return self.run_dataset(bank, stream, [qid])[qid]


REGISTRY = {c.family: c for c in (Single, SC, SCSeq, WVote, Adaptive, Cascade, HWave, ProxyAlloc,
                                  ShortMK, ConfFilter, PVerify)}


def default_grid(bank: Bank, has_certainty: dict) -> list[Strategy]:
    """A sweep that covers every family across every model in the bank."""
    tags = bank.tags_by_size()
    nmax = min(len(v) for mb in bank.models.values() for v in mb.samples.values())
    ns = [n for n in (2, 4, 8, 16, 32, 64) if n <= nmax]
    out: list[Strategy] = []
    for t in tags:
        out.append(Single(model=t))
        out += [SC(model=t, n=n) for n in ns]
        out += [SCSeq(model=t, n=n) for n in ns if n in (4, 16)]
        if has_certainty.get(t):
            out += [WVote(model=t, n=n) for n in ns if n >= 4]
        out += [Adaptive(model=t, tau=tau, max_n=nmax) for tau in (0.9, 0.95)]
        out += [HWave(models=[t], tau=tau, max_waves=mw) for tau in (0.9, 0.95) for mw in (1, 2)]
        if has_certainty.get(t):
            out += [HWave(models=[t], tau=tau, max_waves=2, weighted=True) for tau in (0.9, 0.95)]
        out += [ProxyAlloc(model=t, avg_n=a) for a in (4, 8) if a <= nmax]
        out += [ShortMK(model=t, k=k, m=m) for k in ns if k >= 4 for m in (1, 3) if m < k]
        if has_certainty.get(t):
            out += [ConfFilter(model=t, n=n, keep=kp) for n in ns if n >= 8 for kp in (0.1, 0.5, 0.9)]
        out += [HWave(models=[t], tau=tau, max_waves=2, quorum=True) for tau in (0.8, 0.9, 0.95)]
    pairs = list(zip(tags, tags[1:]))
    if len(tags) > 2:
        pairs.append((tags[0], tags[-1]))
    for s, b in pairs:
        for n1 in (4, 8):
            for tau in (0.6, 0.8, 1.0):
                if n1 <= nmax:
                    out.append(Cascade(models=[s, b], n_small=n1, n_big=min(8, nmax), tau=tau))
        out += [HWave(models=[s, b], tau=tau, max_waves=mw) for tau in (0.9, 0.95) for mw in (1, 2)]
        out += [HWave(models=[s, b], tau=tau, max_waves=1, quorum=True) for tau in (0.9, 0.95)]
    # prefill-verified escalation for every (small, big) pair that has verifier scores
    for b in tags:
        for s in bank.models[b].verify:
            if s in bank.models and s != b:
                out += [PVerify(models=[s, b], n=n, gate_tau=g, mode=md)
                        for n in ("free", 8) for g in (None, 0.9) for md in ("max", "mix")]
    if len(tags) > 2:
        out += [HWave(models=tags, tau=tau, max_waves=1) for tau in (0.9, 0.95)]
    return out


def build_from_config(items: list[dict]) -> list[Strategy]:
    out = []
    for it in items:
        it = dict(it)
        fam = it.pop("family")
        out.append(REGISTRY[fam](**it))
    return out

"""The live FreeWidth engine.

    from freewidth.engine import Engine
    eng = Engine.from_yaml("configs/engine_t4.yaml")
    res = eng.solve("What is 17 * 23?")
    print(res.answer, res.confidence, res.seconds)

Policy `freewidth` (default), per request:
  for each model in the ladder (small -> big):
      run up to `max_waves` waves of the model's free width, each with a live QuorumStop
      -> answer as soon as the pooled votes are confident (Beta posterior >= tau)
      if still unsure and a bigger model exists:
          bigger model prefill-scores the top candidates (P(Yes), no decoding)
          -> accept the best candidate if its score clears `verify_accept`
          otherwise escalate: the bigger model decodes its own waves
"""
from __future__ import annotations

import json
import os
import random
import threading
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field

from ..backends import make_backend
from ..costs import DeviceFit
from ..grading import extract_answer
from ..logs import get_logger
from ..strategies import lead_probability
from .config import EngineConfig, PolicyConfig, SamplingConfig
from .metrics import Metrics
from .runtime import QuorumState, Row, SharedQuorum, WaveResult, generate_wave

log = get_logger("engine")

VERIFY_SYSTEM = "You are a strict grader of math solutions."
VERIFY_TEMPLATE = ("Problem:\n{question}\n\nProposed solution:\n{solution}\n\n"
                   "Is the final answer of the proposed solution correct? Answer Yes or No.")


class DeadlineExceeded(TimeoutError):
    pass


def _is_oom(e: BaseException) -> bool:
    return type(e).__name__ == "OutOfMemoryError" or "out of memory" in str(e).lower()


def _free_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:  # noqa: BLE001 - best-effort cleanup
        log.debug("cuda cache not freed", extra={"error": str(e)})


@dataclass
class WaveTrace:
    model: str
    width: int
    finished: int
    cut: int
    tokens: int
    seconds: float
    joules: float | None
    quorum_hit: bool


@dataclass
class VerifyTrace:
    model: str
    candidates: list
    p_yes: list
    seconds: float
    accepted: str | None


@dataclass
class SolveResult:
    request_id: str
    policy: str
    answer: str | None
    text: str
    confidence: float
    model: str
    escalated: bool = False
    verified: bool = False
    free_form: bool = False
    waves: list = field(default_factory=list)
    verifications: list = field(default_factory=list)
    prompt_tokens: int = 0
    wall_seconds: float | None = None     # set when work overlaps across devices
    notes: dict = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return sum(w.tokens for w in self.waves)

    @property
    def seconds(self) -> float:
        if self.wall_seconds is not None:
            return self.wall_seconds
        return sum(w.seconds for w in self.waves) + sum(v.seconds for v in self.verifications)

    @property
    def joules(self) -> float | None:
        js = [w.joules for w in self.waves]
        return None if not js or any(j is None for j in js) else float(sum(js))

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(tokens=self.tokens, seconds=round(self.seconds, 4),
                 joules=None if self.joules is None else round(self.joules, 2))
        d.pop("text")
        return d


class Engine:
    def __init__(self, cfg: EngineConfig, metrics: Metrics | None = None):
        self.cfg = cfg
        self.metrics = metrics or Metrics()
        self.models: dict = {}
        self.fits: dict = {}
        self.free_widths: dict = {}
        self.replicas: dict = {}          # tag -> list of extra backends (loaded on first use)
        self._replica_errors: dict = {}   # tag -> why replicas could not be loaded
        self._lock = threading.Lock()
        self.loaded = False

    @classmethod
    def from_yaml(cls, path: str, load: bool = True) -> Engine:
        eng = cls(EngineConfig.from_yaml(path))
        if load:
            eng.load()
        return eng

    # ------------------------------------------------------------ lifecycle
    def load(self) -> None:
        for m in self.cfg.ladder:
            t0 = time.time()
            log.info("loading model", extra={"tag": m.tag, "model_id": m.model_id})
            self.models[m.tag] = make_backend(m.model_dump(exclude={"replica_devices"}))
            log.info("loaded", extra={"tag": m.tag, "device": self.models[m.tag].device_name(),
                                      "seconds": round(time.time() - t0, 1)})
            self._calibrate(m.tag)
        self.loaded = True

    def close(self) -> None:
        for b in self.models.values():
            b.close()
        for bs in self.replicas.values():
            for b in bs:
                b.close()
        self.models.clear()
        self.replicas.clear()
        self.loaded = False

    def _typical_L(self) -> int:
        c = self.cfg.calibration
        return c.typical_gen_len or int(0.4 * self.cfg.sampling.max_new_tokens)

    def _calibrate(self, tag: str) -> None:
        c = self.cfg.calibration
        backend = self.models[tag]
        d = os.path.join(c.dir, tag)
        path = os.path.join(d, "device_fit.json")
        device = backend.device_name()
        fit = None
        if os.path.exists(path):
            with open(path) as f:
                j = json.load(f)
            if j.get("device") == device:
                fit = DeviceFit.from_dict(j)
            else:
                log.warning("calibration is for another device; ignoring",
                            extra={"tag": tag, "calibrated": j.get("device"), "current": device})
        if fit is not None and (fit.r2 or 0.0) < c.min_r2:
            log.warning("stored calibration fit is poor; re-measuring",
                        extra={"tag": tag, "fit_r2": fit.r2, "min_r2": c.min_r2})
            fit = None
        if fit is None and c.auto:
            from ..profiler import profile_backend

            for attempt in range(c.max_retries + 1):
                log.info("calibrating free width", extra={"tag": tag, "attempt": attempt + 1})
                profile_backend(backend, {"widths": c.widths, "gen_lens": c.gen_lens,
                                          "prompt_lens": c.prompt_lens, "repeats": c.repeats + attempt,
                                          "warmup": c.warmup},
                                d, typical_L=self._typical_L(), typical_P=c.typical_prompt_len,
                                eps=c.eps, log=lambda m: log.debug(m))
                with open(path) as f:
                    fit = DeviceFit.from_dict(json.load(f))
                if (fit.r2 or 0.0) >= c.min_r2:
                    break
                log.warning("calibration fit below threshold; retrying with more repeats",
                            extra={"tag": tag, "fit_r2": fit.r2, "min_r2": c.min_r2})
            if (fit.r2 or 0.0) < c.min_r2:
                log.warning("calibration still poor after retries; free width may be unreliable",
                            extra={"tag": tag, "fit_r2": fit.r2})
        self.fits[tag] = fit
        pol = self.cfg.policy
        if isinstance(pol.width, int):
            w = pol.width
        elif fit is not None:
            w = fit.free_width(self._typical_L(), c.typical_prompt_len, c.eps)
        else:
            w = 8
            log.warning("no calibration; using width 8", extra={"tag": tag})
        self.free_widths[tag] = max(1, min(int(w), pol.max_width))
        self.metrics.set("free_width", self.free_widths[tag], {"model": tag},
                         help="Calibrated free width per model")
        log.info("free width", extra={"tag": tag, "free_width": self.free_widths[tag],
                                      "fit_r2": getattr(fit, "r2", None)})

    def _get_replicas(self, tag: str) -> list:
        """Extra copies of `tag` on the devices listed in `replica_devices`, loaded once on first
        use. A replica that fails to load (e.g. out of memory) is skipped and reported, and the
        policy runs on the devices that did load."""
        if tag in self.replicas:
            return self.replicas[tag]
        entry = next(m for m in self.cfg.ladder if m.tag == tag)
        primary = self.models[tag]
        out, errors = [], []
        for i, dev in enumerate(entry.replica_devices, 1):
            spec = entry.model_dump(exclude={"replica_devices"})
            spec.update(device=dev, tag=f"{tag}.r{i}")
            t0 = time.time()
            try:
                b = make_backend(spec)
            except Exception as e:  # noqa: BLE001 - report and continue with fewer devices
                errors.append(f"{dev}: {type(e).__name__}: {e}")
                log.warning("replica failed to load", extra={"tag": tag, "device": dev, "error": str(e)})
                _free_cuda_cache()
                continue
            if b.device_name() != primary.device_name():
                log.warning("replica is on a different GPU type; reusing the primary's free width "
                            "anyway", extra={"tag": tag, "primary": primary.device_name(),
                                             "replica": b.device_name()})
            log.info("replica loaded", extra={"tag": b.tag, "device": b.device_name(),
                                              "seconds": round(time.time() - t0, 1)})
            out.append(b)
        self.replicas[tag] = out
        if errors:
            self._replica_errors[tag] = errors
        return out

    def release_replicas(self) -> None:
        """Unload replica copies to give their memory back (they reload on next use)."""
        for bs in self.replicas.values():
            for b in bs:
                b.close()
        self.replicas.clear()
        _free_cuda_cache()

    def info(self) -> dict:
        return {
            "loaded": self.loaded,
            "policy": self.cfg.policy.model_dump(),
            "ladder": [{"tag": t, "device": b.device_name(), "free_width": self.free_widths.get(t),
                        "calibration_r2": getattr(self.fits.get(t), "r2", None),
                        "replicas": [r.device_name() for r in self.replicas.get(t, [])],
                        "replica_errors": self._replica_errors.get(t, [])}
                       for t, b in self.models.items()],
        }

    # --------------------------------------------------------------- solve
    def solve(self, question: str, *, system: str | None = None, raw_prompt: bool = False,
              overrides: dict | None = None, seed: int | None = None,
              deadline: float | None = None, request_id: str | None = None,
              sim_question=None) -> SolveResult:
        """Answer one question. Thread-safe; requests are served one at a time because the
        whole point is to own the device (single tenant)."""
        if not self.loaded:
            raise RuntimeError("engine not loaded")
        ov = overrides or {}
        unknown = set(ov) - set(PolicyConfig.model_fields) - set(SamplingConfig.model_fields)
        if unknown:
            raise ValueError(f"unknown overrides: {sorted(unknown)}")
        # validated copies: a bad value raises pydantic.ValidationError (a ValueError)
        pol = PolicyConfig.model_validate({**self.cfg.policy.model_dump(),
                                           **{k: v for k, v in ov.items() if k in PolicyConfig.model_fields}})
        smp = SamplingConfig.model_validate({**self.cfg.sampling.model_dump(),
                                             **{k: v for k, v in ov.items() if k in SamplingConfig.model_fields}})
        rid = request_id or uuid.uuid4().hex[:16]
        sys_prompt = system or self.cfg.prompt.system
        user = question if raw_prompt else self.cfg.prompt.template.format(question=question)
        rng = random.Random(seed)
        with self._lock:
            t0 = time.perf_counter()
            res = self._run(pol, smp, sys_prompt, user, question, rng, deadline, rid, sim_question)
            self._record(res, time.perf_counter() - t0)
        return res

    def _wave(self, tag, sys_prompt, user, n, smp, rng, quorum, deadline, sim_question) -> WaveResult:
        if deadline is not None and time.time() > deadline:
            raise DeadlineExceeded("request deadline exceeded")
        return generate_wave(self.models[tag], sys_prompt, user, n, temperature=smp.temperature,
                             top_p=smp.top_p, max_new_tokens=smp.max_new_tokens,
                             seed=rng.randrange(2**31), quorum=quorum, question=sim_question)

    def _run_big_sc2(self, res, pol, smp, sys_prompt, user, rng, deadline, sim_q, tag) -> SolveResult:
        """Big model alone, data-parallel over every device that holds a copy of it.

        `sc_n` samples are split evenly over the primary and its replicas; all waves run at the
        same time and share one QuorumStop, so every device stops as soon as the pooled votes are
        confident. This is the fair "just use the big model on all GPUs" baseline."""
        backends = [self.models[tag], *self._get_replicas(tag)]
        k = len(backends)
        sizes = [pol.sc_n // k + (1 if i < pol.sc_n % k else 0) for i in range(k)]
        pairs = [(b, n) for b, n in zip(backends, sizes) if n > 0]
        shared = SharedQuorum(QuorumState(pol.tau, Counter(), pol.min_finished)) if pol.quorum else None
        seeds = [rng.randrange(2**31) for _ in pairs]
        if deadline is not None and time.time() > deadline:
            raise DeadlineExceeded("request deadline exceeded")
        results: list = [None] * len(pairs)
        errors: list = [None] * len(pairs)

        def work(j: int) -> None:
            b, n = pairs[j]
            try:
                results[j] = generate_wave(b, sys_prompt, user, n, temperature=smp.temperature,
                                           top_p=smp.top_p, max_new_tokens=smp.max_new_tokens,
                                           seed=seeds[j], quorum=None, shared=shared, question=sim_q)
            except BaseException as e:  # noqa: BLE001 - re-raised in the calling thread
                errors[j] = e

        t0 = time.perf_counter()
        threads = [threading.Thread(target=work, args=(j,), daemon=True) for j in range(1, len(pairs))]
        for th in threads:
            th.start()
        work(0)
        for th in threads:
            th.join()
        wall = time.perf_counter() - t0
        # a replica that runs out of memory mid-request: redo its share on the primary
        for j, e in enumerate(errors):
            if e is None:
                continue
            if j == 0 or not _is_oom(e):
                raise e
            log.warning("replica out of memory; rerunning its samples on the primary",
                        extra={"tag": tag, "device": pairs[j][0].device_name()})
            _free_cuda_cache()
            res.notes.setdefault("replica_oom", []).append(pairs[j][0].device_name())
            t1 = time.perf_counter()
            results[j] = generate_wave(backends[0], sys_prompt, user, pairs[j][1],
                                       temperature=smp.temperature, top_p=smp.top_p,
                                       max_new_tokens=smp.max_new_tokens, seed=seeds[j],
                                       quorum=None, shared=shared, question=sim_q)
            wall += time.perf_counter() - t1
        rows: list[Row] = []
        for w in results:
            res.waves.append(self._trace(w))
            rows.extend(w.rows)
        res.prompt_tokens = results[0].prompt_tokens
        # simulated waves report modelled time, live waves the real parallel wall-clock
        sim = all(type(b).__name__ == "MockBackend" for b, _ in pairs)
        res.wall_seconds = max(w.wall_s for w in results) if sim else wall
        res.notes.update(devices=[b.device_name() for b, _ in pairs], split=[n for _, n in pairs],
                         quorum_hit=bool(shared and shared.hit))
        if self._replica_errors.get(tag):
            res.notes["replica_errors"] = self._replica_errors[tag]
        c = Counter(r.key for r in rows if r.votes)
        key = c.most_common(1)[0][0] if c else None
        return self._finish(res, rows, key, lead_probability(c) if c else 0.0, tag)

    @staticmethod
    def _trace(w: WaveResult) -> WaveTrace:
        fin = sum(1 for r in w.rows if r.finish == "stop")
        cut = sum(1 for r in w.rows if r.finish == "quorum_cut")
        return WaveTrace(w.model, len(w.rows), fin, cut, w.tokens, round(w.wall_s, 4),
                         None if w.energy_j is None else round(w.energy_j, 2), w.quorum_hit)

    @staticmethod
    def _best_row(rows: list[Row], key: str | None) -> Row | None:
        cands = [r for r in rows if r.votes and r.key == key]
        if cands:
            return min(cands, key=lambda r: r.gen_tokens)  # shortest supporting chain
        fin = [r for r in rows if r.finish != "quorum_cut"]
        if not fin:
            return rows[0] if rows else None
        return max(fin, key=lambda r: (r.self_certainty or float("-inf"), -r.gen_tokens))

    def _finish(self, res: SolveResult, rows: list[Row], key: str | None, conf: float, tag: str) -> SolveResult:
        best = self._best_row(rows, key)
        res.text = best.text if best else ""
        res.answer = extract_answer(best.text) if (best and key is not None) else None
        res.confidence = round(conf, 4)
        res.model = tag
        return res

    def _run(self, pol, smp, sys_prompt, user, question, rng, deadline, rid, sim_q) -> SolveResult:
        ladder = [m.tag for m in self.cfg.ladder]
        res = SolveResult(rid, pol.name, None, "", 0.0, ladder[0])

        if pol.name == "repair":
            from .repair import run_repair

            return run_repair(self, res, pol, smp, sys_prompt, user, question, rng, deadline, sim_q)
        if pol.name == "big_sc":
            tag = ladder[-1]
            w = self._wave(tag, sys_prompt, user, pol.sc_n, smp, rng, None, deadline, sim_q)
            res.waves.append(self._trace(w))
            res.prompt_tokens = w.prompt_tokens
            c = Counter(r.key for r in w.voting_rows)
            key = c.most_common(1)[0][0] if c else None
            return self._finish(res, w.rows, key, lead_probability(c) if c else 0.0, tag)

        if pol.name == "big_sc2":
            return self._run_big_sc2(res, pol, smp, sys_prompt, user, rng, deadline, sim_q, ladder[-1])

        if pol.name == "overlap":
            from .overlap_run import run_overlap

            return run_overlap(self, res, pol, smp, sys_prompt, user, question, rng, deadline, sim_q)

        if pol.name in ("single", "sc"):
            n = 1 if pol.name == "single" else pol.sc_n
            w = self._wave(ladder[0], sys_prompt, user, n, smp, rng, None, deadline, sim_q)
            res.waves.append(self._trace(w))
            res.prompt_tokens = w.prompt_tokens
            c = Counter(r.key for r in w.voting_rows)
            key = c.most_common(1)[0][0] if c else None
            return self._finish(res, w.rows, key, lead_probability(c) if c else 0.0, ladder[0])

        all_rows: list[Row] = []
        level_rows: dict = {}
        for li, tag in enumerate(ladder):
            width = self.free_widths[tag]
            counts: Counter = Counter()
            rows: list[Row] = []
            for wi in range(pol.max_waves):
                q = QuorumState(pol.tau, Counter(counts), pol.min_finished) if pol.quorum else None
                w = self._wave(tag, sys_prompt, user, width, smp, rng, q, deadline, sim_q)
                res.waves.append(self._trace(w))
                res.prompt_tokens = res.prompt_tokens or w.prompt_tokens
                rows += w.rows
                counts.update(r.key for r in w.voting_rows)
                if li > 0:
                    res.escalated = True
                if not counts and wi == 0 and li == 0 and pol.free_form_fallback:
                    res.free_form = True        # nothing checkable to vote on
                    return self._finish(res, rows, None, 0.0, tag)
                if counts and lead_probability(counts) >= pol.tau:
                    key = counts.most_common(1)[0][0]
                    return self._finish(res, rows, key, lead_probability(counts), tag)
            all_rows += rows
            level_rows[tag] = (rows, counts)
            nxt = ladder[li + 1] if li + 1 < len(ladder) else None
            if nxt is None:
                break
            if pol.name == "freewidth" and pol.verify and counts:
                acc = self._verify(res, tag, nxt, question, rows, counts, pol, deadline, sim_q)
                if acc is not None:
                    res.verified = True
                    v = res.verifications[-1]
                    return self._finish(res, rows, acc, v.p_yes[v.candidates.index(acc)], tag)
        # nobody reached tau: plurality of the largest model that voted, ties broken by smaller ones
        tags_with_votes = [t for t in ladder if t in level_rows and level_rows[t][1]]
        if not tags_with_votes:
            return self._finish(res, all_rows, None, 0.0, ladder[-1])
        top_tag = tags_with_votes[-1]
        rows, counts = level_rows[top_tag]
        best = counts.most_common(2)
        key = best[0][0]
        if len(best) > 1 and best[0][1] == best[1][1]:
            small = Counter()
            for t in tags_with_votes[:-1]:
                small.update(level_rows[t][1])
            key = max((best[0][0], best[1][0]), key=lambda k: small.get(k, 0))
        return self._finish(res, rows, key, lead_probability(counts), top_tag)

    def _verify(self, res, small_tag, big_tag, question, rows, counts, pol, deadline,
                sim_q=None) -> str | None:
        if deadline is not None and time.time() > deadline:
            raise DeadlineExceeded("request deadline exceeded")
        cands = [k for k, _ in counts.most_common(pol.top_c)]
        reps = [self._best_row(rows, k) for k in cands]
        users = [VERIFY_TEMPLATE.format(question=question, solution=r.text[-6000:]) for r in reps]
        backend = self.models[big_tag]
        backend_sync = getattr(backend, "_sync", lambda: None)
        backend_sync()
        t0 = time.perf_counter()
        scores = backend.score_yes(VERIFY_SYSTEM, users, batch=len(users),
                                  questions=[sim_q] * len(users) if sim_q is not None else None)
        backend_sync()
        secs = time.perf_counter() - t0
        if hasattr(backend, "_secs"):  # simulated device: report simulated time
            secs = backend._secs(len(users), 1, max(s[1] for s in scores))
        p = [round(s[0], 4) for s in scores]
        total = sum(counts.values())
        mixed = {k: counts[k] / total + p[i] for i, k in enumerate(cands)}
        best = max(mixed, key=mixed.get)
        accepted = best if p[cands.index(best)] >= pol.verify_accept else None
        res.verifications.append(VerifyTrace(big_tag, cands, p, round(secs, 4), accepted))
        return accepted

    def _record(self, res: SolveResult, wall: float) -> None:
        m = self.metrics
        lbl = {"policy": res.policy}
        m.inc("requests_total", 1, lbl, help="Solved requests")
        m.inc("waves_total", len(res.waves), lbl, help="Waves run")
        m.inc("generated_tokens_total", res.tokens, lbl, help="Tokens generated across all samples")
        m.inc("quorum_stops_total", sum(w.quorum_hit for w in res.waves), lbl, help="Waves stopped by QuorumStop")
        m.inc("escalations_total", int(res.escalated), lbl, help="Requests escalated to a bigger model decode")
        m.inc("verifications_total", len(res.verifications), lbl, help="Prefill verification passes")
        m.inc("verified_accepts_total", int(res.verified), lbl, help="Requests answered by prefill verification")
        if res.joules is not None:
            m.inc("energy_joules_total", res.joules, lbl, help="Measured GPU energy")
        m.observe("request_seconds", wall, lbl, help="End-to-end solve time")
        if res.notes.get("trigger"):
            m.inc("big_model_starts_total", 1, {**lbl, "reason": res.notes["trigger"]},
                  help="Overlap: big-model decodes started, by trigger")
        if res.notes.get("accept_reason"):
            m.inc("accepts_total", 1, {**lbl, "reason": res.notes["accept_reason"]},
                  help="Overlap: how the final answer was accepted")
        log.info("solved", extra={"request_id": res.request_id, "policy": res.policy, "model": res.model,
                                  "answer": res.answer, "confidence": res.confidence,
                                  "waves": len(res.waves), "tokens": res.tokens,
                                  "seconds": round(res.seconds, 3), "escalated": res.escalated,
                                  "verified": res.verified, **{f"note_{k}": v for k, v in res.notes.items()}})

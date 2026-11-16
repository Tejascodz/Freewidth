"""Request execution for the `overlap` policy: live (two HF backends) and simulated (mock)."""
from __future__ import annotations

import queue
import threading
import time
from collections import Counter

from ..strategies import lead_probability
from .overlap import OverlapController, hf_cancellable_wave, hf_wave_with_controller
from .runtime import QuorumState, Row, WaveResult

VERIFY_SYSTEM = "You are a strict grader of math solutions."
VERIFY_TEMPLATE = ("Problem:\n{question}\n\nProposed solution:\n{solution}\n\n"
                   "Is the final answer of the proposed solution correct? Answer Yes or No.")


def _controller(pol, width) -> OverlapController:
    return OverlapController(
        width, pol.tau, veto_lo=pol.veto_lo, accept_hi=pol.accept_hi,
        require_verified=pol.require_verified and pol.stream_verify, min_finished=pol.min_finished,
        trigger_min_finished=pol.trigger_min_finished, trigger_lead_prob=pol.trigger_lead_prob,
        runaway_factor=pol.runaway_factor, runaway_min_finished=pol.runaway_min_finished,
        speculative=pol.speculative)


def _check_deadline(deadline):
    from .core import DeadlineExceeded

    if deadline is not None and time.time() > deadline:
        raise DeadlineExceeded("request deadline exceeded")


def _finish(engine, res, ctrl, small_rows, small_tag, big_waves, big_tag, accepted_key, reason):
    """Pick the final answer and fill the result."""
    if accepted_key is not None:
        p = ctrl.best_p(accepted_key)
        conf = p if (reason.startswith("verified") and p is not None) else lead_probability(ctrl.counts)
        res.notes["accept_reason"] = reason
        return engine._finish(res, small_rows, accepted_key, conf, small_tag)
    big_rows = [r for w in big_waves for r in w.rows]
    big_counts = Counter(r.key for r in big_rows if r.votes)
    if big_counts:
        res.escalated = True
        top = big_counts.most_common(2)
        key = top[0][0]
        if len(top) > 1 and top[0][1] == top[1][1]:  # tie on B: B's verification of S's chains breaks it
            key = max((top[0][0], top[1][0]), key=lambda k: (ctrl.best_p(k) or 0.0, ctrl.counts.get(k, 0)))
        res.notes["accept_reason"] = "big_model"
        return engine._finish(res, big_rows, key, lead_probability(big_counts), big_tag)
    # B produced nothing usable: best verified small answer, else small plurality
    scored = [(ctrl.best_p(k), c, k) for k, c in ctrl.counts.items() if ctrl.best_p(k) is not None]
    if scored:
        key = max(scored)[2]
    elif ctrl.counts:
        key = ctrl.counts.most_common(1)[0][0]
    else:
        key = None
    res.notes["accept_reason"] = "fallback"
    return engine._finish(res, small_rows, key, lead_probability(ctrl.counts) if ctrl.counts else 0.0, small_tag)


def _big_waves(engine, big_tag, sys_prompt, user, smp, pol, rng, cancel, out: list, sim_q=None):
    """Big-model waves with QuorumStop; appends WaveResults to `out` until confident or cancelled."""
    from .runtime import generate_wave

    backend = engine.models[big_tag]
    width = engine.free_widths[big_tag]
    counts: Counter = Counter()
    for _ in range(pol.max_waves):
        if cancel.is_set():
            return
        q = QuorumState(pol.tau, Counter(counts), pol.min_finished)
        seed = rng.randrange(2**31)
        if hasattr(backend, "_secs"):  # mock
            w = generate_wave(backend, sys_prompt, user, width, temperature=smp.temperature, top_p=smp.top_p,
                              max_new_tokens=smp.max_new_tokens, seed=seed, quorum=q, question=sim_q)
        else:
            w = hf_cancellable_wave(backend, sys_prompt, user, width, temperature=smp.temperature,
                                    top_p=smp.top_p, max_new_tokens=smp.max_new_tokens, seed=seed,
                                    quorum=q, cancel=cancel)
        out.append(w)
        counts.update(r.key for r in w.voting_rows)
        if counts and lead_probability(counts) >= pol.tau:
            return


# ---------------------------------------------------------------------------
# live path
# ---------------------------------------------------------------------------

def _run_live(engine, res, pol, smp, sys_prompt, user, question, rng, deadline):
    small_tag, big_tag = engine.cfg.ladder[0].tag, engine.cfg.ladder[1].tag
    small, big = engine.models[small_tag], engine.models[big_tag]
    width = engine.free_widths[small_tag]
    ctrl = _controller(pol, width)
    big_lock = threading.Lock()          # verification and big decoding never share B at once
    stop_verify = threading.Event()
    cancel_big = threading.Event()
    jobs: queue.Queue = queue.Queue()
    results: queue.Queue = queue.Queue()
    big_waves: list = []
    state = {"big_thread": None, "verified": 0, "verify_s": 0.0, "pending": 0}
    plock = threading.Lock()

    def verifier():
        while True:
            item = jobs.get()
            if item is None:
                return
            key, text = item
            try:
                if stop_verify.is_set():
                    continue
                with big_lock:
                    if stop_verify.is_set():
                        continue
                    t0 = time.perf_counter()
                    p, _ = big.score_yes(VERIFY_SYSTEM, [VERIFY_TEMPLATE.format(
                        question=question, solution=text[-6000:])], batch=1)[0]
                    state["verify_s"] += time.perf_counter() - t0
                results.put((key, p))
                state["verified"] += 1
            finally:
                with plock:
                    state["pending"] -= 1

    def on_finish(row, key, text):
        if pol.stream_verify and key is not None and not stop_verify.is_set():
            with plock:
                state["pending"] += 1
            jobs.put((key, text))

    def poll():
        while True:
            try:
                key, p = results.get_nowait()
            except queue.Empty:
                return
            ctrl.on_verified(key, p)

    def start_big(reason):
        if state["big_thread"] is not None:
            return
        res.notes["trigger"] = reason
        res.notes["trigger_s"] = round(time.perf_counter() - t_start, 3)
        stop_verify.set()

        def run():
            with big_lock:
                _big_waves(engine, big_tag, sys_prompt, user, smp, pol, rng, cancel_big, big_waves)

        th = threading.Thread(target=run, daemon=True)
        state["big_thread"] = th
        th.start()

    vt = threading.Thread(target=verifier, daemon=True)
    vt.start()
    t_start = time.perf_counter()
    try:
        _check_deadline(deadline)
        rows, P, wall, joules, decision = hf_wave_with_controller(
            small, sys_prompt, user, width, temperature=smp.temperature, top_p=smp.top_p,
            max_new_tokens=smp.max_new_tokens, seed=rng.randrange(2**31), ctrl=ctrl,
            on_finish=on_finish, on_trigger=start_big if pol.speculative else (lambda r: None), poll=poll)
        res.prompt_tokens = P
        res.waves.append(engine._trace(WaveResult(rows, P, wall, joules, decision is not None, None, small_tag)))
        accepted, reason = (decision.accept, decision.reason) if decision else (None, "")
        if accepted is None and state["big_thread"] is None:
            # wave over: give in-flight verifications a bounded chance to land, then look again
            t_wait = time.perf_counter()
            while state["pending"] > 0 and time.perf_counter() - t_wait < pol.verify_wait_s:
                time.sleep(0.05)
            poll()
            d = ctrl.end_of_wave()
            accepted, reason = d.accept, d.reason
        if accepted is not None:
            cancel_big.set()
        else:
            start_big("sequential_fallback" if state["big_thread"] is None else res.notes.get("trigger", ""))
        if state["big_thread"] is not None:
            remaining = None if deadline is None else max(0.0, deadline - time.time())
            state["big_thread"].join(timeout=remaining)
            if state["big_thread"].is_alive():
                cancel_big.set()
                state["big_thread"].join(timeout=30)
                _check_deadline(deadline)
        for w in big_waves:
            tr = engine._trace(w)
            if accepted is not None:
                tr.model = f"{tr.model} (cancelled)"
            res.waves.append(tr)
        res.wall_seconds = round(time.perf_counter() - t_start, 4)
    finally:
        stop_verify.set()
        jobs.put(None)
    res.notes["verified_rows"] = state["verified"]
    res.notes["verify_seconds"] = round(state["verify_s"], 3)
    res.notes["cut_rows"] = len(ctrl.cut)
    res.verified = accepted is not None and reason.startswith("verified")
    return _finish(engine, res, ctrl, rows, small_tag,
                   big_waves if accepted is None else [], big_tag, accepted, reason)


# ---------------------------------------------------------------------------
# simulated path (mock backends): event simulation on the fitted device curves
# ---------------------------------------------------------------------------

def _run_sim(engine, res, pol, smp, sys_prompt, user, question, rng, deadline, sim_q):
    small_tag, big_tag = engine.cfg.ladder[0].tag, engine.cfg.ladder[1].tag
    small, big = engine.models[small_tag], engine.models[big_tag]
    width = engine.free_widths[small_tag]
    ctrl = _controller(pol, width)
    _check_deadline(deadline)
    g = small.generate(sys_prompt, user, width, temperature=smp.temperature, top_p=smp.top_p,
                       max_new_tokens=smp.max_new_tokens, seed=rng.randrange(2**31), question=sim_q)
    P = g.prompt_tokens
    lens = list(g.gen_tokens)
    step_s = max(1e-6, small._secs(width, 2, P) - small._secs(width, 1, P))
    verify_queue: list = []          # (finish_step, key, text)
    busy_until = 0.0                 # verifier busy until this step (fractional)
    done_verifs: list = []           # (complete_step, key, p)
    trigger_step = None
    accepted, reason, end_step = None, "", max(lens)
    order = sorted(range(width), key=lambda i: (lens[i], i))
    for step in range(1, max(lens) + 1):
        for i in order:
            if lens[i] == step and i not in ctrl.cut:
                key = ctrl.on_finish(i, g.texts[i], step, step)
                if pol.stream_verify and key is not None and trigger_step is None:
                    verify_queue.append((step, key, g.texts[i]))
        # verifier: single worker, FIFO, prefill time on the big model's curve
        while verify_queue:
            fstep, key, text = verify_queue[0]
            start = max(busy_until, fstep)
            if start > step:
                break
            ptxt = VERIFY_TEMPLATE.format(question=question, solution=text)
            p, ptok = big.score_yes(VERIFY_SYSTEM, [ptxt], questions=[sim_q])[0]
            dur = big._secs(1, 1, ptok) / step_s
            busy_until = start + dur
            done_verifs.append((busy_until, key, p))
            verify_queue.pop(0)
        for c in [v for v in done_verifs if v[0] <= step]:
            ctrl.on_verified(c[1], c[2])
            done_verifs.remove(c)
        d = ctrl.decide(step)
        if d.trigger and pol.speculative and trigger_step is None:
            trigger_step = step
            verify_queue.clear()
            res.notes["trigger"] = d.trigger_reason
        if d.accept is not None:
            accepted, reason, end_step = d.accept, d.reason, step
            break
        if all(i in ctrl.finished or i in ctrl.cut for i in range(width)):
            end_step = step
            break
    if accepted is None and trigger_step is None:
        for c in done_verifs:
            ctrl.on_verified(c[1], c[2])
        d = ctrl.end_of_wave()
        accepted, reason = d.accept, d.reason
    rows = []
    for i in range(width):
        if i in ctrl.finished:
            k, L, _ = ctrl.finished[i]
            rows.append(Row(g.texts[i], k, L, "stop"))
        else:
            rows.append(Row(g.texts[i], None, min(lens[i], end_step), "quorum_cut"))
    small_s = small._secs(width, end_step, P)
    small_j = small_s * small._watts
    res.prompt_tokens = P
    res.waves.append(engine._trace(WaveResult(rows, P, small_s, small_j, accepted is not None, end_step, small_tag)))
    big_waves: list = []
    if accepted is None or trigger_step is not None:
        start_s = small._secs(width, trigger_step, P) if trigger_step is not None else small_s
        if trigger_step is None:
            res.notes["trigger"] = "sequential_fallback"
        _big_waves(engine, big_tag, sys_prompt, user, smp, pol, rng, threading.Event(), big_waves, sim_q)
        big_s = sum(w.wall_s for w in big_waves)
        if accepted is not None:   # speculative big run cancelled when the small model accepted
            ran = max(0.0, small_s - start_s)
            frac = min(1.0, ran / big_s) if big_s > 0 else 0.0
            for w in big_waves:
                tr = engine._trace(w)
                tr.model, tr.seconds = f"{tr.model} (cancelled)", round(tr.seconds * frac, 4)
                tr.joules = None if tr.joules is None else round(tr.joules * frac, 2)
                res.waves.append(tr)
            res.wall_seconds = round(small_s, 4)
            big_waves = []
        else:
            for w in big_waves:
                res.waves.append(engine._trace(w))
            res.wall_seconds = round(max(small_s, start_s + big_s), 4)
    else:
        res.wall_seconds = round(small_s, 4)
    res.notes["cut_rows"] = len(ctrl.cut)
    res.notes["verified_rows"] = sum(len(v) for v in ctrl.pyes.values())
    res.verified = accepted is not None and reason.startswith("verified")
    return _finish(engine, res, ctrl, rows, small_tag, big_waves, big_tag, accepted, reason)


def run_overlap(engine, res, pol, smp, sys_prompt, user, question, rng, deadline, sim_q):
    if len(engine.cfg.ladder) < 2:
        raise ValueError("policy 'overlap' needs a ladder of at least two models")
    small = engine.models[engine.cfg.ladder[0].tag]
    if hasattr(small, "_secs"):
        return _run_sim(engine, res, pol, smp, sys_prompt, user, question, rng, deadline, sim_q)
    return _run_live(engine, res, pol, smp, sys_prompt, user, question, rng, deadline)


__all__ = ["run_overlap"]

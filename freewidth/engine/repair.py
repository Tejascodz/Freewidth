"""Read-Locate-Fork (policy `repair`): the big model reads once and writes at most one token.

When the small model S is not confident after its free-width waves:

1. READ (one forward pass on the big model B per candidate). The candidate is shown to B as
   someone else's solution in a grading prompt ("Problem / Proposed solution / Is the final
   answer correct?"). One prefill gives both
     - the verdict P(Yes) at the last position, and
     - B's next-token distribution at every solution token (where the reasoning broke).
2. LOCATE. The flaw is the first solution token (after a short warm-up) that B gives
   probability < `flaw_p` while B's own preferred token has probability >= `flaw_conf`.
3. FORK. B's preferred token replaces S's token at the flaw. S continues from
   "question + solution[:flaw] + B's token" in a full free-width wave with QuorumStop, so
   the repair costs about one small-model wave.
4. If a candidate is verified (P(Yes) >= accept_hi) it is accepted without repair; if the
   repaired wave reaches a quorum and B does not veto its lead, that answer is accepted;
   otherwise B decodes as a last resort.

Requires S and B to share a tokenizer (true within one model family, e.g. Qwen3.5). If the
vocabularies differ, the policy falls back to verification + escalation.

Closest prior work: StepBack (first-error localization + suffix resampling, same model or an
external scorer), "Reasoning Can Be Restored by Correcting a Few Decision Tokens" (token-level
disagreement between a base and a stronger model), SpecAgent (strong model prefill-verifies a
weak draft and repairs the first rejected step by itself). Here the verdict and the location
come from the same pass, the stronger model writes one token, and the repair is a parallel
free-width wave of the weaker model.
"""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass

from ..grading import answer_key, extract_answer
from ..logs import get_logger

log = get_logger("repair")



@dataclass
class ReadResult:
    p_yes: float
    flaw: int | None          # index into the solution token ids
    fix_token: int | None     # B's preferred token at the flaw
    solution_ids: list
    prefix_ids: list          # generation prompt ids (shared tokenizer)
    seconds: float
    tokens: int


READ_SYSTEM = "You are a strict grader of math solutions."
READ_TEMPLATE = ("Problem:\n{question}\n\nProposed solution:\n{solution}\n\n"
                 "Is the final answer of the proposed solution correct? Answer Yes or No.")
_SLOT = "\u2063SOLUTION\u2063"


def shared_tokenizer(small, big) -> bool:
    try:
        a, b = small.tok, big.tok
        probe = "x = 17 * 23, so \\boxed{391}"
        return len(a) == len(b) and a.encode(probe) == b.encode(probe)
    except Exception:
        return False


def _third_party_pieces(big, question: str) -> tuple[str, str]:
    """Chat-formatted grading prompt split around the solution slot.

    The solution is shown as someone else's work. Framing it as the model's own assistant
    turn (0.4.0) made P(Yes) collapse to ~0.99 for every candidate on the T4 run: the
    self-correction blind spot (Tsui 2025) showing up in a verifier.
    """
    text = big._chat_text(READ_SYSTEM, READ_TEMPLATE.format(question=question, solution=_SLOT), False)
    pre, post = text.split(_SLOT, 1)
    return pre, post


def hf_read(big, system: str, user: str, solution_text: str, *, flaw_p: float = 0.05,
            flaw_conf: float = 0.5, warmup_tokens: int = 8, max_solution_tokens: int = 2048,
            question: str | None = None) -> ReadResult:
    """One forward pass of B over a third-party grading prompt -> verdict + first flawed token.

    The verdict is P(Yes) at the last position. The flaw comes from B's next-token
    distribution over the solution span inside the same prompt."""
    t = big._torch
    tok = big.tok
    gen_prefix_ids = tok(big._chat_text(system, user, False), add_special_tokens=False)["input_ids"]
    pre, post = _third_party_pieces(big, question if question is not None else user)
    pre_ids = tok(pre, add_special_tokens=False)["input_ids"]
    sol_ids = tok(solution_text, add_special_tokens=False)["input_ids"][:max_solution_tokens]
    post_ids = tok(post, add_special_tokens=False)["input_ids"]
    ids = pre_ids + sol_ids + post_ids
    x = t.tensor([ids], device=big.model.device)
    big._sync()
    t0 = time.perf_counter()
    with t.inference_mode():
        logits = big.model(input_ids=x).logits[0]          # (T, V)
        last = t.softmax(logits[-1].float(), dim=-1)
        yes = _first_ids(tok, ("Yes", " Yes", "yes"))
        no = _first_ids(tok, ("No", " No", "no"))
        py, pn = float(last[yes].sum()), float(last[no].sum())
        p_yes = py / (py + pn) if py + pn > 0 else 0.5
        flaw, fix = None, None
        start = len(pre_ids)
        for c0 in range(0, len(sol_ids), 256):
            c1 = min(len(sol_ids), c0 + 256)
            lp = t.log_softmax(logits[start - 1 + c0: start - 1 + c1].float(), dim=-1)
            tgt = t.tensor(sol_ids[c0:c1], device=lp.device)
            p_tok = lp.gather(1, tgt[:, None])[:, 0].exp()
            p_top, top = lp.max(dim=-1)
            p_top = p_top.exp()
            bad = (p_tok < flaw_p) & (p_top >= flaw_conf) & (top != tgt)
            idx = [i for i in t.nonzero(bad).flatten().tolist() if c0 + i >= warmup_tokens]
            if idx:
                flaw, fix = c0 + idx[0], int(top[idx[0]])
                break
    big._sync()
    return ReadResult(p_yes, flaw, fix, sol_ids, gen_prefix_ids, time.perf_counter() - t0, len(ids))


def _first_ids(tok, words) -> list:
    out = set()
    for w in words:
        e = tok.encode(w, add_special_tokens=False)
        if e:
            out.add(e[0])
    return sorted(out)


def hf_fork_wave(small, read: ReadResult, n: int, *, temperature: float, top_p: float,
                 max_new_tokens: int, seed: int | None, quorum):
    """S continues from question + solution[:flaw] + B's token, n rows, live QuorumStop."""
    from transformers import StoppingCriteriaList

    from ..backends.hf import _StatsRecorder
    from .runtime import Row, WaveResult, _QuorumCriteria

    t = small._torch
    head = read.solution_ids[: read.flaw] + [read.fix_token]
    ids = read.prefix_ids + head
    x = t.tensor([ids], device=small.model.device)
    mask = t.ones_like(x)
    P = len(ids)
    eos = small._eos_ids()
    head_text = small.tok.decode(head, skip_special_tokens=True)
    q = _QuorumCriteria(_PrefixTok(small.tok, head_text), P, eos, quorum)
    if seed is not None:
        small._seed(seed)
    small._sync()
    small.meter.start()
    t0 = time.perf_counter()
    with t.inference_mode():
        out = small.model.generate(
            input_ids=x, attention_mask=mask, do_sample=temperature > 0, temperature=max(temperature, 1e-5),
            top_p=top_p, top_k=0, max_new_tokens=max(16, max_new_tokens - len(head)), num_return_sequences=n,
            logits_processor=[_StatsRecorder()], pad_token_id=small.tok.pad_token_id,
            stopping_criteria=StoppingCriteriaList([q]))
    small._sync()
    wall = time.perf_counter() - t0
    joules = small.meter.stop()
    rows = []
    for gen in out[:, P:].cpu().tolist():
        L = next((k + 1 for k, tk in enumerate(gen) if tk in eos), None)
        if L is None:
            L, fin = len(gen), ("quorum_cut" if q.hit else "length")
        else:
            fin = "stop"
        text = head_text + small.tok.decode(gen[:L], skip_special_tokens=True)
        rows.append(Row(text, answer_key(extract_answer(text)) if fin == "stop" else None, L, fin))
    return WaveResult(rows, P, wall, joules, q.hit, None, small.tag)


class _PrefixTok:
    """Tokenizer shim so the quorum criterion extracts answers from head + continuation."""

    def __init__(self, tok, head_text: str):
        self.tok, self.head = tok, head_text

    def decode(self, ids, skip_special_tokens=True):
        return self.head + self.tok.decode(ids, skip_special_tokens=skip_special_tokens)


# ---------------------------------------------------------------------------
# policy driver
# ---------------------------------------------------------------------------

def run_repair(engine, res, pol, smp, sys_prompt, user, question, rng, deadline, sim_q):
    from .core import DeadlineExceeded
    from .overlap_run import _big_waves
    from .runtime import QuorumState, generate_wave

    if len(engine.cfg.ladder) < 2:
        raise ValueError("policy 'repair' needs a ladder of at least two models")
    small_tag, big_tag = engine.cfg.ladder[0].tag, engine.cfg.ladder[-1].tag
    small, big = engine.models[small_tag], engine.models[big_tag]
    mock = hasattr(small, "_secs")
    width = engine.free_widths[small_tag]
    t_start = time.perf_counter()
    sim_secs = 0.0

    def check():
        if deadline is not None and time.time() > deadline:
            raise DeadlineExceeded("request deadline exceeded")

    # 1. small-model waves with QuorumStop (as in hwave)
    from ..strategies import lead_probability

    counts: Counter = Counter()
    rows: list = []
    for _ in range(pol.max_waves):
        check()
        w = generate_wave(small, sys_prompt, user, width, temperature=smp.temperature, top_p=smp.top_p,
                          max_new_tokens=smp.max_new_tokens, seed=rng.randrange(2**31),
                          quorum=QuorumState(pol.tau, Counter(counts), pol.min_finished), question=sim_q)
        res.waves.append(engine._trace(w))
        res.prompt_tokens = res.prompt_tokens or w.prompt_tokens
        sim_secs += w.wall_s
        rows += w.rows
        counts.update(r.key for r in w.voting_rows)
        if counts and lead_probability(counts) >= pol.tau:
            key = counts.most_common(1)[0][0]
            res.notes["accept_reason"] = "quorum"
            if mock:
                res.wall_seconds = round(sim_secs, 4)
            return engine._finish(res, rows, key, lead_probability(counts), small_tag)
    if not counts:
        res.notes["accept_reason"] = "free_form"
        res.free_form = True
        return engine._finish(res, rows, None, 0.0, small_tag)

    # 2. READ each top candidate once: verdict + flaw from the same pass
    can_fork = mock or shared_tokenizer(small, big)
    cands = [k for k, _ in counts.most_common(pol.top_c)]
    reads = {}
    for k in cands:
        check()
        rep = engine._best_row(rows, k)
        if mock:
            p, ptok = big.score_yes("", [f"Proposed solution:{rep.text}"], questions=[sim_q])[0]
            flaw = None if p >= 0.5 else max(8, rep.gen_tokens // 3)
            r = ReadResult(p, flaw, 0 if flaw is not None else None, [], [], big._secs(1, 1, ptok + rep.gen_tokens),
                           ptok + rep.gen_tokens)
        else:
            r = hf_read(big, sys_prompt, user, rep.text, flaw_p=pol.flaw_p, flaw_conf=pol.flaw_conf,
                        question=question)
        reads[k] = (r, rep)
        sim_secs += r.seconds
    from .core import VerifyTrace

    res.verifications.append(VerifyTrace(big_tag, cands, [round(reads[k][0].p_yes, 4) for k in cands],
                                         round(sum(reads[k][0].seconds for k in cands), 4), None))
    total = sum(counts.values())
    best = max(cands, key=lambda k: counts[k] / total + reads[k][0].p_yes)
    if reads[best][0].p_yes >= pol.accept_hi:
        res.verified = True
        res.verifications[-1].accepted = best
        res.notes["accept_reason"] = "verified"
        if mock:
            res.wall_seconds = round(sim_secs, 4)
        return engine._finish(res, rows, best, reads[best][0].p_yes, small_tag)

    # 3. LOCATE + FORK: repair the most-voted candidate that has a located flaw
    fork_from = next((k for k in cands if reads[k][0].flaw is not None), None)
    if can_fork and fork_from is not None:
        check()
        r, rep = reads[fork_from]
        res.notes["repaired_from"] = fork_from
        res.notes["flaw_token"] = r.flaw
        q = QuorumState(pol.tau, Counter(), pol.min_finished)
        if mock:
            fw = _mock_fork(engine, small, big, sys_prompt, user, width, smp, rng, q, sim_q, r.flaw)
        else:
            fw = hf_fork_wave(small, r, width, temperature=smp.temperature, top_p=smp.top_p,
                              max_new_tokens=smp.max_new_tokens, seed=rng.randrange(2**31), quorum=q)
        tr = engine._trace(fw)
        tr.model = f"{tr.model} (fork)"
        res.waves.append(tr)
        sim_secs += fw.wall_s
        fc = Counter(x.key for x in fw.voting_rows)
        if fc and lead_probability(fc) >= pol.tau:
            lead = fc.most_common(1)[0][0]
            rep2 = engine._best_row(fw.rows, lead)
            if mock:
                p2, _ = big.score_yes("", [f"Proposed solution:{rep2.text}"], questions=[sim_q])[0]
                s2 = big._secs(1, 1, rep2.gen_tokens + 100)
            else:
                rr = hf_read(big, sys_prompt, user, rep2.text, flaw_p=pol.flaw_p, flaw_conf=pol.flaw_conf,
                             question=question)
                p2, s2 = rr.p_yes, rr.seconds
            sim_secs += s2
            res.verifications.append(VerifyTrace(big_tag, [lead], [round(p2, 4)], round(s2, 4),
                                                 lead if p2 >= pol.veto_lo else None))
            if p2 >= pol.veto_lo:
                res.notes["accept_reason"] = "repaired"
                if mock:
                    res.wall_seconds = round(sim_secs, 4)
                return engine._finish(res, fw.rows, lead, lead_probability(fc), small_tag)
        res.notes["repair_failed"] = True
    # 4. last resort: B decodes
    check()
    import threading

    big_ws: list = []
    _big_waves(engine, big_tag, sys_prompt, user, smp, pol, rng, threading.Event(), big_ws, sim_q)
    for w in big_ws:
        res.waves.append(engine._trace(w))
        sim_secs += w.wall_s
    res.escalated = True
    if mock:
        res.wall_seconds = round(sim_secs, 4)
    else:
        res.wall_seconds = round(time.perf_counter() - t_start, 4)
    big_rows = [x for w in big_ws for x in w.rows]
    bc = Counter(x.key for x in big_rows if x.votes)
    res.notes["accept_reason"] = "big_model"
    if bc:
        return engine._finish(res, big_rows, bc.most_common(1)[0][0], lead_probability(bc), big_tag)
    return engine._finish(res, rows, best, 0.0, small_tag)


def _mock_fork(engine, small, big, sys_prompt, user, width, smp, rng, quorum, sim_q, flaw):
    """Simulated repair: a fork from a located flaw is right more often than a fresh sample,
    by an amount tied to the big model's skill, and shorter (the prefix is reused)."""
    import random

    from .runtime import mock_generate_wave

    w = mock_generate_wave(small, sys_prompt, user, width, temperature=smp.temperature, top_p=smp.top_p,
                           max_new_tokens=smp.max_new_tokens, seed=rng.randrange(2**31), quorum=quorum,
                           question=sim_q)
    gold = sim_q.answer if sim_q is not None else None
    r = random.Random(rng.randrange(2**31))
    for row in w.rows:
        if row.finish == "stop" and gold is not None and row.key != answer_key(gold) and r.random() < 0.6 * big.skill:
            row.key = answer_key(gold)
            row.text = row.text.rsplit("\\boxed", 1)[0] + "\\boxed{" + gold + "}"
        row.gen_tokens = max(4, row.gen_tokens - (flaw or 0))
    w.wall_s = small._secs(width, max(r_.gen_tokens for r_ in w.rows), w.prompt_tokens + (flaw or 0))
    return w

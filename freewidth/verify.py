"""Prefill-only verification pass for Prefill-Verified Escalation.

For each (small, big) pair, the big model reads every stored small-model solution
once and records P(Yes | "is this final answer correct?"). Replay then decides which
candidates a strategy would actually have verified and prices only those passes.
"""
from __future__ import annotations

import json
import os

from .backends import make_backend
from .config import model_dir
from .data import load_questions

DEFAULT_SYSTEM = "You are a strict grader of math solutions."
DEFAULT_TEMPLATE = (
    "Problem:\n{question}\n\nProposed solution:\n{solution}\n\n"
    "Is the final answer of the proposed solution correct? Answer Yes or No."
)


def _rows(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass


def verify_pair(cfg: dict, small: str, big: str, log=print) -> None:
    vc = cfg.get("verify", {})
    sp = os.path.join(model_dir(cfg, small), "samples.jsonl")
    rows = [r for r in _rows(sp) if "text" in r]
    if not rows:
        raise RuntimeError(f"{small}: samples have no text; set bank.keep_text: true and rebuild")
    out_path = os.path.join(model_dir(cfg, big), f"verify_{small}.jsonl")
    done = set()
    if os.path.exists(out_path):
        done = {(r["qid"], r["sid"]) for r in _rows(out_path)}
    qs = {q.qid: q for q in load_questions(cfg["dataset"])}
    todo = [r for r in rows if (r["qid"], f"{r['wave']}-{r['i']}") not in done and r["qid"] in qs]
    if not todo:
        log(f"[verify {small}->{big}] complete")
        return
    mcfg = next(m for m in cfg["models"] if m["tag"] == big)
    backend = make_backend(mcfg)
    system = vc.get("system", DEFAULT_SYSTEM)
    template = vc.get("template", DEFAULT_TEMPLATE)
    max_chars = int(vc.get("max_chars", 6000))
    batch = int(vc.get("batch", 8))
    log(f"[verify {small}->{big}] scoring {len(todo)} solutions")
    with open(out_path, "a") as f:
        for k in range(0, len(todo), batch * 4):
            chunk = todo[k : k + batch * 4]
            users = [template.format(question=qs[r["qid"]].question,
                                     solution=r["text"][-max_chars:]) for r in chunk]
            scores = backend.score_yes(system, users, batch, [qs[r["qid"]] for r in chunk])
            for r, (p, ptok) in zip(chunk, scores):
                f.write(json.dumps({"qid": r["qid"], "sid": f"{r['wave']}-{r['i']}",
                                    "p_yes": p, "prompt_tokens": ptok,
                                    "small_correct": r["correct"]}) + "\n")
            f.flush()
            if (k // (batch * 4)) % 25 == 0:
                log(f"[verify {small}->{big}] {min(k + batch * 4, len(todo))}/{len(todo)}")
    backend.close()


def run(cfg: dict, log=print) -> None:
    pairs = cfg.get("verify", {}).get("pairs") or []
    for p in pairs:
        verify_pair(cfg, p["small"], p["big"], log)

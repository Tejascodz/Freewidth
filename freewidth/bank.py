"""Generate a reusable sample bank: N samples per (model, question), drawn in waves.

Every strategy is later evaluated by *replaying* this bank, so the expensive
GPU part runs once. Generation is resumable: re-running skips finished work,
which matters on Kaggle/Colab where sessions end without warning.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter

from .backends import make_backend
from .config import model_dir, public_model_cfg
from .data import load_questions, render_prompt
from .grading import extract_answer, is_equiv


def _existing_counts(path: str) -> Counter:
    c: Counter = Counter()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    try:
                        c[json.loads(line)["qid"]] += 1
                    except (json.JSONDecodeError, KeyError):
                        pass  # truncated last line from a killed session
    return c


def _next_wave_idx(path: str) -> Counter:
    c: Counter = Counter()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line)
                        c[r["qid"]] = max(c[r["qid"]], r["wave"] + 1)
                    except (json.JSONDecodeError, KeyError):
                        pass
    return c


def build_bank_for_model(cfg: dict, mcfg: dict, questions, deadline: float | None, log=print) -> bool:
    tag = mcfg["tag"]
    out = model_dir(cfg, tag)
    sp, wp = os.path.join(out, "samples.jsonl"), os.path.join(out, "waves.jsonl")
    target = int(mcfg.get("samples_per_question", cfg["bank"]["samples_per_question"]))
    width = int(mcfg.get("wave_width", cfg["bank"]["wave_width"]))
    have = _existing_counts(sp)
    todo = [q for q in questions if have[q.qid] < target]
    if not todo:
        log(f"[{tag}] bank complete ({len(questions)} questions x {target})")
        return True
    backend = make_backend(mcfg)
    s = {**cfg["sampling"], **(mcfg.get("sampling") or {})}
    meta = {"tag": tag, "model": public_model_cfg(mcfg), "device": backend.device_name(),
            "measures_device": backend.measures_device, "sampling": s,
            "dataset": cfg["dataset"], "prompt": cfg.get("prompt", {})}
    spec = backend.spec()
    if spec is not None:
        meta["spec"] = spec.to_dict()
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    wave_no = _next_wave_idx(wp)
    t_start, done = time.time(), 0
    log(f"[{tag}] {len(todo)} questions to fill on {backend.device_name()}")
    with open(sp, "a") as fs, open(wp, "a") as fw:
        for q in todo:
            system, user = render_prompt(q, cfg.get("prompt"))
            while have[q.qid] < target:
                if deadline and time.time() > deadline:
                    log(f"[{tag}] time budget reached; progress saved, re-run to resume")
                    backend.close()
                    return False
                n = min(width, target - have[q.qid])
                widx = wave_no[q.qid]
                seed = int(hashlib.sha256(f"{cfg['bank']['seed']}|{tag}|{q.qid}|{widx}".encode())
                           .hexdigest()[:8], 16) & 0x7FFFFFFF
                r = backend.generate(system, user, n, temperature=s["temperature"],
                                     top_p=s["top_p"], max_new_tokens=s["max_new_tokens"],
                                     seed=seed, question=q)
                for i, text in enumerate(r.texts):
                    ans = extract_answer(text)
                    row = {
                        "qid": q.qid, "wave": widx, "i": i, "answer": ans,
                        "correct": bool(is_equiv(ans, q.answer)),
                        "gen_tokens": int(r.gen_tokens[i]), "prompt_tokens": int(r.prompt_tokens),
                        "finish": r.finish[i] if i < len(r.finish) else "",
                        "self_certainty": r.self_certainty[i] if i < len(r.self_certainty) else None,
                        "entropy": r.entropy[i] if i < len(r.entropy) else None,
                        "mean_logprob": r.mean_logprob[i] if i < len(r.mean_logprob) else None,
                    }
                    if mcfg.get("keep_text", cfg["bank"].get("keep_text", False)):
                        row["text"] = text
                    fs.write(json.dumps(row) + "\n")
                fw.write(json.dumps({"qid": q.qid, "wave": widx, "width": len(r.texts),
                                     "max_gen": max(r.gen_tokens) if r.gen_tokens else 0,
                                     "prompt_tokens": r.prompt_tokens, "wall_s": r.wall_s,
                                     "energy_j": r.energy_j}) + "\n")
                fs.flush()
                fw.flush()
                have[q.qid] += len(r.texts)
                wave_no[q.qid] += 1
            done += 1
            if done % 10 == 0 or done == len(todo):
                el = time.time() - t_start
                eta = el / done * (len(todo) - done)
                log(f"[{tag}] {done}/{len(todo)} questions  elapsed {el/60:.1f}m  eta {eta/60:.1f}m")
    backend.close()
    return True


def run(cfg: dict, only: list[str] | None = None, max_hours: float | None = None, log=print) -> bool:
    questions = load_questions(cfg["dataset"])
    os.makedirs(cfg["output_dir"], exist_ok=True)
    with open(os.path.join(cfg["output_dir"], "questions.jsonl"), "w") as f:
        for q in questions:
            f.write(json.dumps({"qid": q.qid, "answer": q.answer, "meta": q.meta}) + "\n")
    deadline = time.time() + max_hours * 3600 if max_hours else None
    ok = True
    for m in cfg["models"]:
        if only and m["tag"] not in only:
            continue
        ok = build_bank_for_model(cfg, m, questions, deadline, log) and ok
        if deadline and time.time() > deadline:
            return False
    return ok

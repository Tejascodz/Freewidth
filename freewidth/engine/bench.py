"""Live benchmark: run the engine's policies on a dataset and measure real cost.

    freewidth bench --config configs/engine_t4.yaml --dataset gsm8k --limit 100 \
        --policies single,sc,hwave,freewidth

Unlike `analyze` (offline replay priced by a fitted curve), every number here is
measured on the device: wall seconds, NVML joules, generated tokens. Use it to check
that the replay's predictions hold live.
"""
from __future__ import annotations

import csv
import json
import os
import statistics as st
import time

from ..data import load_questions
from ..grading import is_equiv
from ..logs import get_logger
from .core import Engine

log = get_logger("bench")


def run(config: str, dataset: str, limit: int, policies: list[str], out_dir: str,
        seed: int = 0, split: str = "test", max_hours: float | None = None) -> str:
    """Run `policies` in order on the same questions. With `max_hours`, no new question is
    started after the budget is spent: the unfinished policy is reported as partial, later
    policies are skipped, and every finished row and the report are still written. This lets
    a time-limited job (e.g. a 12-hour Kaggle commit) end cleanly instead of being killed."""
    deadline = time.time() + max_hours * 3600 if max_hours else None
    eng = Engine.from_yaml(config)
    qs = load_questions({"name": dataset, "split": split, "limit": limit, "seed": seed})
    os.makedirs(out_dir, exist_ok=True)
    info = eng.info()
    with open(os.path.join(out_dir, "engine_info.json"), "w") as f:
        json.dump(info, f, indent=2, default=str)
    rows_path = os.path.join(out_dir, "bench_rows.jsonl")
    big = info["ladder"][-1]["tag"] if info.get("ladder") else None
    summary = []
    out_of_time = False
    try:
        with open(rows_path, "w") as f:
            for pol in policies:
                if out_of_time:
                    log.warning("skipped: time budget spent", extra={"policy": pol})
                    continue
                t0 = time.time()
                recs = []
                for i, q in enumerate(qs):
                    if deadline is not None and time.time() > deadline:
                        out_of_time = True
                        log.warning("time budget spent; stopping", extra={"policy": pol, "done": i, "of": len(qs)})
                        break
                    res = eng.solve(q.question, overrides={"name": pol}, seed=seed * 100003 + i,
                                    sim_question=q)
                    ok = is_equiv(res.answer, q.answer)
                    rec = {"policy": pol, "qid": q.qid, "correct": ok, **res.to_dict()}
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    recs.append(rec)
                    if (i + 1) % 10 == 0:
                        log.info("progress", extra={"policy": pol, "done": i + 1, "of": len(qs)})
                if not recs:
                    continue
                js = [r["joules"] for r in recs if r["joules"] is not None]
                summary.append({
                    "policy": pol, "n": len(recs), "complete": len(recs) == len(qs),
                    "accuracy": sum(r["correct"] for r in recs) / len(recs),
                    "mean_seconds": st.mean(r["seconds"] for r in recs),
                    "median_seconds": st.median(r["seconds"] for r in recs),
                    "p90_seconds": sorted(r["seconds"] for r in recs)[int(0.9 * (len(recs) - 1))],
                    "big_model_rate": sum(_big_decoded(r, big) for r in recs) / len(recs),
                    "mean_joules": st.mean(js) if len(js) == len(recs) else None,
                    "mean_tokens": st.mean(r["tokens"] for r in recs),
                    "escalation_rate": sum(r["escalated"] for r in recs) / len(recs),
                    "verified_rate": sum(r["verified"] for r in recs) / len(recs),
                    "wall_minutes": (time.time() - t0) / 60,
                })
                log.info("policy done", extra=summary[-1])
                eng.release_replicas()   # free GPU memory before the next policy
                _write_report(out_dir, dataset, qs, info, summary)  # keep partial results
    finally:
        eng.close()
    return _write_report(out_dir, dataset, qs, info, summary)

def _big_decoded(rec: dict, big: str | None) -> bool:
    """Did the largest model (or one of its replicas) generate tokens for this question?"""
    if big is None:
        return False
    return any(w.get("model", "").split(".r")[0] == big and w.get("tokens", 0) > 0
               for w in rec.get("waves", []))


def _write_report(out_dir, dataset, qs, info, summary):
    if not summary:
        return None
    with open(os.path.join(out_dir, "bench_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    calib = "; ".join(f"`{r['tag']}` free width {r['free_width']} (R² {r['calibration_r2']:.3f}, {r['device']})"
                      if r["calibration_r2"] is not None else f"`{r['tag']}` free width {r['free_width']}"
                      for r in info["ladder"])
    lines = [f"# Live benchmark: {dataset} ({len(qs)} questions)", "", f"Calibration: {calib}", "",
             "| policy | n | accuracy | mean s | median s | p90 s | mean J | mean tokens | escalated | largest model decoded | verified |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summary:
        mj = f"{s['mean_joules']:.1f}" if s["mean_joules"] is not None else "n/a"
        n = f"{s['n']}" + ("" if s.get("complete", True) else " (partial: time budget)")
        lines.append(f"| {s['policy']} | {n} | {s['accuracy']:.3f} | {s['mean_seconds']:.2f} | {s['median_seconds']:.2f} | "
                     f"{s['p90_seconds']:.2f} | {mj} | {s['mean_tokens']:.0f} | {s['escalation_rate']:.2f} | "
                     f"{s['big_model_rate']:.2f} | {s['verified_rate']:.2f} |")
    path = os.path.join(out_dir, "BENCH.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path

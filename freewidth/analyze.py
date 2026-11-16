"""Replay every strategy over the bank, price it three ways, and write the report."""
from __future__ import annotations

import csv
import json
import math
import os
import random
import statistics as st

from .pareto import (
    best_under_budget,
    fit_saturating,
    jaccard,
    kendall_tau,
    pareto_front,
    plot_device_curves,
    plot_frontiers,
)
from .replay import Bank, Stream, is_key_correct, load_bank, price_trace
from .strategies import build_from_config, default_grid

COSTS = ["flops", "kinetic", "device_s", "device_j"]


def evaluate(bank: Bank, strategies, trials: int, intensity: float, static_batch: bool,
             n_boot: int, log=print) -> list[dict]:
    rows = []
    qids = bank.qids
    for si, strat in enumerate(strategies):
        if any(m not in bank.models for m in strat.models()):
            continue
        per_q_correct = {q: 0.0 for q in qids}
        sums = {k: 0.0 for k in COSTS + ["tokens", "rounds", "samples"]}
        trial_accs = []
        for t in range(trials):
            stream = Stream(bank, t)
            traces = strat.run_dataset(bank, stream, qids)
            correct = 0
            for q in qids:
                tr = traces[q]
                ok = is_key_correct(bank, q, tr.key)
                correct += ok
                per_q_correct[q] += ok
                c = price_trace(bank, q, tr, intensity, static_batch)
                for k in sums:
                    sums[k] += c[k]
            trial_accs.append(correct / len(qids))
        nq = len(qids) * trials
        row = {"strategy": strat.name, "family": strat.family,
               "models": "+".join(strat.models()), "acc": st.mean(trial_accs)}
        row.update({k: v / nq for k, v in sums.items()})
        lo, hi = _bootstrap([per_q_correct[q] / trials for q in qids], n_boot)
        row["acc_lo"], row["acc_hi"] = lo, hi
        rows.append(row)
        if (si + 1) % 10 == 0:
            log(f"  evaluated {si + 1}/{len(strategies)} strategies")
    return rows


def _bootstrap(vals: list[float], n: int, alpha: float = 0.05) -> tuple[float, float]:
    if not vals or n <= 0:
        return float("nan"), float("nan")
    rng = random.Random(0)
    k = len(vals)
    means = sorted(sum(vals[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return means[int(alpha / 2 * n)], means[min(n - 1, int((1 - alpha / 2) * n))]


def disagreement(rows: list[dict], quantiles: list[float]) -> dict:
    """How differently the cost models rank the same set of strategies."""
    out: dict = {"kendall_tau": {}, "pareto_jaccard": {}, "budget_winners": []}
    keys = [k for k in COSTS if any(not math.isnan(r[k]) for r in rows)]
    fronts = {}
    for k in keys:
        fronts[k] = {rows[i]["strategy"] for i in pareto_front([(r[k], r["acc"]) for r in rows])}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            out["kendall_tau"][f"{a}~{b}"] = kendall_tau([r[a] for r in rows], [r[b] for r in rows])
            out["pareto_jaccard"][f"{a}~{b}"] = jaccard(fronts[a], fronts[b])
    # Winners at matched budgets: pick budgets as quantiles of each metric's range
    for k in keys:
        vals = sorted(r[k] for r in rows if not math.isnan(r[k]))
        for qt in quantiles:
            b = vals[min(len(vals) - 1, int(qt * (len(vals) - 1)))]
            w = best_under_budget(rows, k, b)
            if w:
                out["budget_winners"].append({"cost": k, "quantile": qt, "budget": b,
                                              "winner": w["strategy"], "acc": w["acc"]})
    out["fronts"] = {k: sorted(v) for k, v in fronts.items()}
    return out


def iso_accuracy_overrun(rows: list[dict], guide: str, deploy: str) -> dict | None:
    """If you pick strategies off the `guide` frontier but pay `deploy` cost, how much more do
    you pay than the cheapest strategy (under `deploy`) that is at least as accurate?"""
    if not any(not math.isnan(r[guide]) for r in rows) or not any(not math.isnan(r[deploy]) for r in rows):
        return None
    front = [rows[i] for i in pareto_front([(r[guide], r["acc"]) for r in rows])]
    out = []
    for r in front:
        if math.isnan(r[deploy]):
            continue
        ok = [x[deploy] for x in rows if x["acc"] >= r["acc"] - 1e-9 and not math.isnan(x[deploy])]
        best = min(ok)
        out.append({"strategy": r["strategy"], "acc": r["acc"], "overrun": r[deploy] / best if best > 0 else 1.0})
    if not out:
        return None
    ovs = sorted(o["overrun"] for o in out)
    worst = max(out, key=lambda o: o["overrun"])
    return {"guide": guide, "deploy": deploy, "median": ovs[len(ovs) // 2], "max": worst["overrun"],
            "worst_strategy": worst["strategy"], "worst_acc": worst["acc"], "points": out}


def scaling_fits(rows: list[dict], cost_key: str) -> list[dict]:
    groups: dict = {}
    for r in rows:
        if r["family"] in ("sc", "wvote", "sc_seq"):
            groups.setdefault((r["family"], r["models"]), []).append(r)
    out = []
    for (fam, m), rs in sorted(groups.items()):
        fit = fit_saturating([r[cost_key] for r in rs], [r["acc"] for r in rs])
        if fit:
            A, B, al, c0 = fit
            out.append({"family": fam, "models": m, "cost": cost_key, "A_ceiling": A,
                        "B": B, "alpha": al, "c0": c0})
    return out


def _fmt(x, k):
    if isinstance(x, float) and math.isnan(x):
        return "n/a"
    if k in ("flops", "kinetic"):
        return f"{x:.3g}"
    if k in ("device_s", "device_j"):
        return f"{x:.3f}"
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def write_report(run_dir: str, bank: Bank, rows: list[dict], dis: dict, fits: list[dict],
                 overruns: list[dict] | None = None) -> str:
    lines = [f"# Results: {os.path.basename(os.path.abspath(run_dir))}", ""]
    lines.append(f"Questions: {len(bank.qids)}. Models: " + ", ".join(
        f"`{t}` ({bank.models[t].size / 1e9:.2f}B" + (f", free width {bank.models[t].free_width}" if
        bank.models[t].free_width else "") + ")" for t in bank.tags_by_size()) + ".")
    lines += ["", "## Do the cost models agree?", "",
              "Kendall tau between the cost each model assigns to the same strategies "
              "(1 = identical ordering). Pareto Jaccard = overlap of frontier sets.", "",
              "| pair | Kendall tau | Pareto Jaccard |", "|---|---|---|"]
    for pair, tau in dis["kendall_tau"].items():
        lines.append(f"| {pair} | {tau:.3f} | {dis['pareto_jaccard'][pair]:.3f} |")
    if overruns:
        lines += ["", "## Cost of trusting the wrong frontier (iso-accuracy overrun)", "",
                  "Pick strategies from the *guide* frontier, pay the *deploy* cost. Overrun = deploy cost / "
                  "cheapest deploy cost reaching the same accuracy (1.0 = no loss).", "",
                  "| guide -> deploy | median overrun | worst overrun | worst strategy |", "|---|---|---|---|"]
        for o in overruns:
            lines.append(f"| {o['guide']} -> {o['deploy']} | {o['median']:.2f}x | {o['max']:.2f}x | "
                         f"`{o['worst_strategy']}` (acc {o['worst_acc']:.3f}) |")
    lines += ["", "## Best strategy at matched budgets", "",
              "| cost model | budget quantile | budget | winner | acc |", "|---|---|---|---|---|"]
    for w in dis["budget_winners"]:
        lines.append(f"| {w['cost']} | {w['quantile']} | {_fmt(w['budget'], w['cost'])} | "
                     f"`{w['winner']}` | {w['acc']:.3f} |")
    for k in dis["fronts"]:
        lines += ["", f"## Pareto frontier under `{k}`", "",
                  "| strategy | acc (95% CI) | " + k + " | rounds |", "|---|---|---|---|"]
        fr = [r for r in rows if r["strategy"] in set(dis["fronts"][k])]
        for r in sorted(fr, key=lambda r: r[k]):
            lines.append(f"| `{r['strategy']}` | {r['acc']:.3f} ({r['acc_lo']:.3f}-{r['acc_hi']:.3f}) | "
                         f"{_fmt(r[k], k)} | {r['rounds']:.2f} |")
    if fits:
        lines += ["", "## Saturating fits  acc(C) = A - B (C/c0)^-alpha", "",
                  "| family | models | cost | A (ceiling) | alpha |", "|---|---|---|---|---|"]
        for f in fits:
            lines.append(f"| {f['family']} | {f['models']} | {f['cost']} | {f['A_ceiling']:.3f} | {f['alpha']:.3f} |")
    lines += ["", "Figures: `figures/frontiers.png`, `figures/device_curves.png`.", ""]
    path = os.path.join(run_dir, "analysis", "REPORT.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


def run(cfg: dict, log=print) -> str:
    run_dir = cfg["output_dir"]
    a = cfg["analysis"]
    bank = load_bank(run_dir, cfg, a.get("min_samples"))
    log(f"loaded bank: {len(bank.qids)} questions, models={bank.tags_by_size()}")
    has_sc = {t: any(s.sc is not None for v in mb.samples.values() for s in v)
              for t, mb in bank.models.items()}
    strategies = build_from_config(a["strategies"]) if a.get("strategies") else default_grid(bank, has_sc)
    log(f"replaying {len(strategies)} strategies x {a['trials']} trials")
    rows = evaluate(bank, strategies, a["trials"], a["kinetic_intensity"], a["static_batch"],
                    a["bootstrap"], log)
    out = os.path.join(run_dir, "analysis")
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    with open(os.path.join(out, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    dis = disagreement(rows, a["budget_quantiles"])
    fits = []
    for k in ("flops", "device_s"):
        if any(not math.isnan(r[k]) for r in rows):
            fits += scaling_fits(rows, k)
    overruns = [o for o in (iso_accuracy_overrun(rows, g, d) for g, d in
                            (("flops", "device_s"), ("kinetic", "device_s"), ("device_s", "flops"),
                             ("device_s", "kinetic"))) if o]
    with open(os.path.join(out, "disagreement.json"), "w") as f:
        json.dump({**dis, "fits": fits, "overruns": overruns}, f, indent=2)
    plot_frontiers(rows, COSTS, os.path.join(out, "figures", "frontiers.png"),
                   title=os.path.basename(os.path.abspath(run_dir)))
    profiles = {}
    for t in bank.models:
        p = os.path.join(run_dir, "models", t, "profile.jsonl")
        if os.path.exists(p):
            with open(p) as f:
                profiles[t] = [json.loads(line) for line in f if line.strip()]
    plot_device_curves(profiles, os.path.join(out, "figures", "device_curves.png"))
    path = write_report(run_dir, bank, rows, dis, fits, overruns)
    log(f"report: {path}")
    return path

"""Statistical report for live benchmark runs.

    python -m freewidth report results/run_a results/run_b --out docs/report_ab

Reads one or more `bench_rows.jsonl` files (or the directories that hold them) and writes
`REPORT.md`, `report.json` and `accuracy_latency.png`. Everything a claim needs is computed
here rather than by hand:

- accuracy with a Wilson 95% interval; mean / median / p90 seconds; mean joules and tokens;
  how often the largest model decoded;
- every pair of policies on the questions both answered: wins / losses, exact McNemar p,
  Holm-adjusted p across all pairs, paired-bootstrap 95% intervals for the accuracy and
  mean-seconds differences, and the number of questions an 80%-power McNemar test would need
  for the observed difference;
- per-policy breakdowns by the engine's own notes (accept reason, escalation trigger).

The bootstrap is seeded, so the same rows always give the same report.
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.stats import binomtest, norm

BOOT = 10_000


# ------------------------------------------------------------------- loading
def _rows_path(p: str) -> str:
    if os.path.isdir(p):
        p = os.path.join(p, "bench_rows.jsonl")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return p


def load_runs(paths: list[str]) -> list[dict]:
    """All rows from all runs. A policy that appears in more than one run is renamed
    `policy@run` (run = the directory name), so repeats are compared, not merged."""
    per_run = []
    for p in paths:
        fp = _rows_path(p)
        run = os.path.basename(os.path.dirname(os.path.abspath(fp)))
        with open(fp) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        per_run.append((run, rows))
    seen = Counter(pol for _, rows in per_run for pol in {r["policy"] for r in rows})
    out = []
    for run, rows in per_run:
        for r in rows:
            r = dict(r)
            if seen[r["policy"]] > 1:
                r["policy"] = f"{r['policy']}@{run}"
            r["_run"] = run
            out.append(r)
    return out


def _size(tag: str) -> float:
    m = re.findall(r"(\d+(?:\.\d+)?)\s*[bB]", tag)
    return float(m[-1]) if m else 0.0


def largest_model(rows: list[dict], ladder: list[str] | None = None) -> str | None:
    if ladder:
        return ladder[-1]
    tags = {w["model"].split(".r")[0] for r in rows for w in r.get("waves", [])}
    return max(tags, key=_size) if tags else None


# --------------------------------------------------------------- statistics
def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar test on the discordant pairs."""
    return 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue)


def holm(ps: list[float]) -> list[float]:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj = [0.0] * len(ps)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def mcnemar_n_needed(p10: float, p01: float, alpha: float = 0.05, power: float = 0.8) -> float | None:
    """Questions needed for a two-sided McNemar test to detect the observed paired difference
    (Connor 1987). None when the observed difference is zero."""
    d = p10 - p01
    pd = p10 + p01
    if d == 0 or pd == 0:
        return None
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    return float(((za * math.sqrt(pd) + zb * math.sqrt(pd - d * d)) ** 2) / (d * d))


def paired_boot(x: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    idx = rng.integers(0, len(x), size=(BOOT, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ------------------------------------------------------------------ summary
@dataclass
class PolicyStats:
    policy: str
    n: int
    accuracy: float
    acc_lo: float
    acc_hi: float
    mean_s: float
    median_s: float
    p90_s: float
    mean_j: float | None
    mean_tokens: float
    big_decode_rate: float | None


def _p90(xs) -> float:
    xs = sorted(xs)
    return xs[int(0.9 * (len(xs) - 1))]


def summarize(rows: list[dict], big: str | None) -> list[PolicyStats]:
    by = defaultdict(list)
    for r in rows:
        by[r["policy"]].append(r)
    out = []
    for pol, rs in by.items():
        k = sum(bool(r["correct"]) for r in rs)
        lo, hi = wilson(k, len(rs))
        secs = [r["seconds"] for r in rs]
        js = [r.get("joules") for r in rs]
        bigrate = None
        if big:
            bigrate = sum(any(w["model"].split(".r")[0] == big and w["tokens"] > 0
                              for w in r.get("waves", [])) for r in rs) / len(rs)
        out.append(PolicyStats(pol, len(rs), k / len(rs), lo, hi, float(np.mean(secs)),
                               float(np.median(secs)), _p90(secs),
                               None if any(j is None for j in js) else float(np.mean(js)),
                               float(np.mean([r.get("tokens", 0) for r in rs])), bigrate))
    return sorted(out, key=lambda s: (-s.accuracy, s.median_s))


@dataclass
class PairStats:
    a: str
    b: str
    n: int
    b_wins: int
    b_losses: int
    p: float
    p_holm: float
    d_acc: float
    d_acc_lo: float
    d_acc_hi: float
    d_mean_s: float
    d_s_lo: float
    d_s_hi: float
    n_for_power: float | None
    primary: bool = False


def pairwise(rows: list[dict], order: list[str], seed: int = 0,
             primary: list[tuple[str, str]] | None = None) -> list[PairStats]:
    """All pairs on their common questions. Holm correction is applied within a family: the
    pre-declared `primary` comparisons form one family, everything else is exploratory and is
    corrected as a second family."""
    idx = {(r["policy"], r["qid"]): r for r in rows}
    qids = defaultdict(set)
    for r in rows:
        qids[r["policy"]].add(r["qid"])
    rng = np.random.default_rng(seed)
    out = []
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            common = sorted(qids[a] & qids[b])
            if not common:
                continue
            ca = np.array([bool(idx[(a, q)]["correct"]) for q in common], dtype=float)
            cb = np.array([bool(idx[(b, q)]["correct"]) for q in common], dtype=float)
            sa = np.array([idx[(a, q)]["seconds"] for q in common])
            sb = np.array([idx[(b, q)]["seconds"] for q in common])
            wins = int(((cb == 1) & (ca == 0)).sum())
            losses = int(((cb == 0) & (ca == 1)).sum())
            n = len(common)
            dacc = cb - ca
            ds = sb - sa
            alo, ahi = paired_boot(dacc, rng)
            slo, shi = paired_boot(ds, rng)
            out.append(PairStats(a, b, n, wins, losses, mcnemar_p(wins, losses), 1.0,
                                 float(dacc.mean()), alo, ahi, float(ds.mean()), slo, shi,
                                 mcnemar_n_needed(wins / n, losses / n)))
    prim = {frozenset(x) for x in (primary or [])}
    for ps in out:
        ps.primary = frozenset((ps.a, ps.b)) in prim
    missing = prim - {frozenset((p.a, p.b)) for p in out}
    if missing:
        raise ValueError(f"primary comparisons not found in the runs: {[tuple(m) for m in missing]}")
    for fam in (True, False):
        group = [p for p in out if p.primary is fam]
        for ps, adj in zip(group, holm([p.p for p in group])):
            ps.p_holm = adj
    return sorted(out, key=lambda p: not p.primary)


def breakdowns(rows: list[dict]) -> dict:
    """Per policy: accuracy and mean seconds grouped by the engine's own notes."""
    out: dict = {}
    for key in ("accept_reason", "trigger"):
        by = defaultdict(lambda: defaultdict(list))
        for r in rows:
            v = (r.get("notes") or {}).get(key)
            if v is not None or key == "trigger" and "accept_reason" in (r.get("notes") or {}):
                by[r["policy"]][v or "none"].append(r)
        for pol, groups in by.items():
            if len(groups) < 2 and key == "trigger":
                continue
            out.setdefault(pol, {})[key] = {
                g: {"n": len(rs), "accuracy": sum(bool(r["correct"]) for r in rs) / len(rs),
                    "mean_s": float(np.mean([r["seconds"] for r in rs]))}
                for g, rs in sorted(groups.items(), key=lambda kv: -len(kv[1]))}
    return out


# ---------------------------------------------------------------- rendering
_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def plot(stats: list[PolicyStats], path: str, title: str) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    for i, s in enumerate(sorted(stats, key=lambda s: s.policy)):
        col = _PALETTE[i % len(_PALETTE)] if i < len(_PALETTE) else "#888888"
        ax.errorbar(s.median_s, s.accuracy, yerr=[[s.accuracy - s.acc_lo], [s.acc_hi - s.accuracy]],
                    fmt="o", ms=8, color=col, ecolor=col, elinewidth=2, capsize=0, zorder=3,
                    markeredgecolor="white", markeredgewidth=2)
        ax.annotate(s.policy, (s.median_s, s.accuracy), xytext=(8, 4), textcoords="offset points",
                    fontsize=9, color="#333333")
    ax.set_xlabel("median seconds per question")
    ax.set_ylabel("accuracy (95% Wilson interval)")
    ax.set_title(title, fontsize=11, loc="left")
    ax.grid(color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#bbbbbb")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _f(x, nd=1, pct=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{100 * x:.{nd}f}" if pct else f"{x:,.{nd}f}"


def render_md(stats, pairs, brk, runs, title) -> str:
    L = [f"# {title}", "", f"Runs: {', '.join(sorted(runs))}. Generated by `python -m freewidth report`.", "",
         "## Policies", "",
         "| policy | n | accuracy % (95% CI) | mean s | median s | p90 s | mean J | mean tokens | largest model decoded |",
         "|---|---|---|---|---|---|---|---|---|"]
    for s in stats:
        L.append(f"| {s.policy} | {s.n} | {_f(s.accuracy, 1, True)} ({_f(s.acc_lo, 1, True)}–{_f(s.acc_hi, 1, True)}) "
                 f"| {_f(s.mean_s)} | {_f(s.median_s)} | {_f(s.p90_s)} | {_f(s.mean_j, 0)} | {_f(s.mean_tokens, 0)} "
                 f"| {_f(s.big_decode_rate, 0, True) + '%' if s.big_decode_rate is not None else '—'} |")
    L += ["", "## Paired comparisons", "",
          "Same questions only. *B wins / loses*: questions B got right and A wrong, and the reverse. "
          "*p*: exact McNemar; *Holm p*: adjusted within its family (the comparisons declared primary "
          "before the run, or all the exploratory ones). Intervals are "
          f"paired bootstrap ({BOOT:,} resamples, seeded). *n for 80% power*: questions a McNemar test "
          "would need to detect the observed accuracy difference at α = 0.05.", "",
          "| A → B | family | n | B wins / loses | p | Holm p | Δ accuracy, points (95% CI) | Δ mean s (95% CI) | n for 80% power |",
          "|---|---|---|---|---|---|---|---|---|"]
    for p in pairs:
        need = "—" if p.n_for_power is None else f"{math.ceil(p.n_for_power):,}"
        fam = "**primary**" if p.primary else "exploratory"
        L.append(f"| {p.a} → {p.b} | {fam} | {p.n} | {p.b_wins} / {p.b_losses} | {p.p:.3g} | {p.p_holm:.3g} "
                 f"| {100 * p.d_acc:+.1f} ({100 * p.d_acc_lo:+.1f} to {100 * p.d_acc_hi:+.1f}) "
                 f"| {p.d_mean_s:+.1f} ({p.d_s_lo:+.1f} to {p.d_s_hi:+.1f}) | {need} |")
    if brk:
        L += ["", "## Breakdowns by engine decision", ""]
        for pol, kinds in brk.items():
            for kind, groups in kinds.items():
                L += [f"**{pol}**, by `{kind}`:", "", "| group | n | accuracy % | mean s |", "|---|---|---|---|"]
                for g, v in groups.items():
                    L.append(f"| {g} | {v['n']} | {_f(v['accuracy'], 1, True)} | {_f(v['mean_s'])} |")
                L.append("")
    L += ["", "![accuracy vs latency](accuracy_latency.png)", ""]
    return "\n".join(L)


def build(paths: list[str], out_dir: str, title: str | None = None, ladder: list[str] | None = None,
          seed: int = 0, primary: list[tuple[str, str]] | None = None) -> str:
    rows = load_runs(paths)
    if not rows:
        raise ValueError("no rows found")
    big = largest_model(rows, ladder)
    stats = summarize(rows, big)
    order = [s.policy for s in stats]
    pairs = pairwise(rows, order, seed, primary)
    brk = breakdowns(rows)
    runs = {r["_run"] for r in rows}
    title = title or "Live benchmark report"
    os.makedirs(out_dir, exist_ok=True)
    plot(stats, os.path.join(out_dir, "accuracy_latency.png"), title)
    md = render_md(stats, pairs, brk, runs, title)
    with open(os.path.join(out_dir, "REPORT.md"), "w") as f:
        f.write(md)
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump({"largest_model": big, "policies": [s.__dict__ for s in stats],
                   "pairs": [p.__dict__ for p in pairs], "breakdowns": brk}, f, indent=2)
    return os.path.join(out_dir, "REPORT.md")

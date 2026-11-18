"""Regenerate the README results figure from the committed per-question rows.

    python scripts/make_figures.py
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from freewidth.report import wilson  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = [  # (rows file, policy in that file, label, colour)
    ("results/t4_gsm8k_E1_2026-11-16/bench_rows.jsonl", "big_sc2", "4B alone, both GPUs", "#2a78d6"),
    ("results/t4_gsm8k_H10/bench_rows.jsonl", "overlap", "overlap (default)", "#1baf7a"),
    ("results/t4_gsm8k_E1_2026-11-16/bench_rows.jsonl", "overlap", "overlap, quorum_accept=true", "#eb6834"),
    ("results/t4_gsm8k_E1_2026-11-16/bench_rows.jsonl", "repair", "repair", "#eda100"),
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    for path, policy, label, colour in RUNS:
        rows = [json.loads(x) for x in (ROOT / path).read_text().splitlines() if x.strip()]
        rows = [r for r in rows if r["policy"] == policy]
        k = sum(bool(r["correct"]) for r in rows)
        acc = k / len(rows)
        lo, hi = wilson(k, len(rows))
        med = st.median(r["seconds"] for r in rows)
        ax.errorbar(med, acc, yerr=[[acc - lo], [hi - acc]], fmt="o", ms=8, color=colour, ecolor=colour,
                    elinewidth=2, capsize=0, markeredgecolor="white", markeredgewidth=2, zorder=3)
        ax.annotate(label, (med, acc), xytext=(9, 3), textcoords="offset points", fontsize=9, color="#333333")
    ax.set_xlabel("median seconds per question")
    ax.set_ylabel("accuracy (95% Wilson interval)")
    ax.set_title("GSM8K, 300 questions, 2 x T4", fontsize=11, loc="left")
    ax.set_xlim(15, 47)
    ax.grid(color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#bbbbbb")
    fig.tight_layout()
    out = ROOT / "docs/figures/results_gsm8k.png"
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main()

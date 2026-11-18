"""Regenerate the README figures from committed data.

    python scripts/make_figures.py

Writes docs/figures/hero.png (free width and headline result) and docs/figures/results_gsm8k.png.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator  # noqa: E402

from freewidth.costs import DeviceFit  # noqa: E402
from freewidth.report import wilson  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, GREEN, AMBER, GREY, INK, MUTED = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                                "#9aa0a6", "#202124", "#5f6368")
CALIB = "results/t4_gsm8k_2026-11-15/calibration"
E1 = "results/t4_gsm8k_E1_2026-11-16/bench_rows.jsonl"
H10 = "results/t4_gsm8k_H10/bench_rows.jsonl"
RUNS = [  # (rows file, policy, label, colour, emphasised)
    (E1, "big_sc2", "4B alone, both GPUs", BLUE, True),
    (H10, "overlap", "FreeWidth overlap", GREEN, True),
    (E1, "overlap", "overlap, quorum_accept=true", ORANGE, False),
    (E1, "repair", "repair", AMBER, False),
]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#c4c7c5",
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 11.5,
})


def _style(ax):
    ax.grid(color="#ebebeb", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _policy_point(path, policy):
    rows = [json.loads(x) for x in (ROOT / path).read_text().splitlines() if x.strip()]
    rows = [r for r in rows if r["policy"] == policy]
    k = sum(bool(r["correct"]) for r in rows)
    lo, hi = wilson(k, len(rows))
    return k / len(rows), lo, hi, st.median(r["seconds"] for r in rows)


def _free_width_panel(ax):
    widths = [1, 2, 4, 8, 16, 32, 64]
    ax.axhspan(0.0, 1.25, color=GREEN, alpha=0.08, lw=0)
    ax.text(1.08, 0.86, "within 1.25× of one sample", color=GREEN, fontsize=8.5, ha="left")
    ax.plot(widths, widths, ls="--", color=GREY, lw=1.4)
    ax.text(1.62, 2.9, "if cost grew with FLOPs", color=MUTED, fontsize=8.5, rotation=63,
            ha="center", va="center")
    for model, label, colour, dy in (("q35-0.8b", "Qwen3.5-0.8B", GREEN, 1.6), ("q35-4b", "Qwen3.5-4B", BLUE, 0.55)):
        f = json.loads((ROOT / CALIB / model / "device_fit.json").read_text())
        fit, L, P = DeviceFit.from_dict(f), f["typical_gen_len"], f["typical_prompt_len"]
        ys = [fit.seconds(w, L, P) / fit.seconds(1, L, P) for w in widths]
        ax.plot(widths, ys, "o-", color=colour, lw=2.2, ms=6, markeredgecolor="white", markeredgewidth=1.5,
                label=label, zorder=3)
        fw = f["free_width"]
        y = fit.seconds(fw, L, P) / fit.seconds(1, L, P)
        ax.plot([fw], [y], "o", ms=12, mfc="none", mec=colour, mew=2, zorder=4)
        ax.annotate(f"free width {fw}", (fw, y), xytext=(fw * dy, y + (0.75 if dy < 1 else 0.38)), fontsize=9,
                    color=colour, fontweight="bold", ha="center",
                    arrowprops=dict(arrowstyle="-", color=colour, lw=1))
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(widths))
    ax.set_xticklabels([str(w) for w in widths])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(FixedLocator([1, 2, 3, 4, 5]))
    ax.set_yticklabels(["1×", "2×", "3×", "4×", "5×"])
    ax.set_ylim(0.8, 5.0)
    ax.set_xlabel("parallel samples per question")
    ax.set_ylabel("wall time relative to one sample")
    ax.text(64, 4.75, "measured on a T4, fitted cost model", color=MUTED, fontsize=8, ha="right")
    ax.set_title("Parallel samples are near-free, up to the free width", loc="left")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    _style(ax)


def _results_panel(ax, emphasise=True):
    for path, policy, label, colour, main in RUNS:
        acc, lo, hi, med = _policy_point(path, policy)
        alpha = 1.0 if (main or not emphasise) else 0.45
        ax.errorbar(med, acc * 100, yerr=[[100 * (acc - lo)], [100 * (hi - acc)]], fmt="o",
                    ms=9 if main else 7, color=colour, ecolor=colour, elinewidth=2.2, capsize=0, alpha=alpha,
                    markeredgecolor="white", markeredgewidth=1.8, zorder=3)
        right = med > 35
        ax.annotate(label, (med, acc * 100), xytext=(10, -18) if right else (10, -3 if main else 2),
                    textcoords="offset points", ha="left",
                    fontsize=9.5 if main else 8.5, color=INK if main else MUTED,
                    fontweight="bold" if main else "normal", alpha=1.0 if main else 0.9)
    if emphasise:
        ax.annotate("", xy=(21.6, 94.6), xytext=(39.4, 95.6),
                    arrowprops=dict(arrowstyle="->", color=INK, lw=1.2, shrinkA=6, shrinkB=6))
        ax.text(30.5, 96.3, "1.9× lower median latency\n44% less energy, −2.0 pts (n.s.)",
                ha="center", fontsize=9, color=INK)
    ax.set_xlim(15, 56)
    ax.set_ylim(78, 99.5)
    ax.set_xlabel("median seconds per question")
    ax.set_ylabel("accuracy, % (95% CI)")
    ax.set_title("GSM8K, 300 questions, 2 × T4", loc="left")
    _style(ax)


def hero():
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.4, 4.6), dpi=170, gridspec_kw={"wspace": 0.28})
    _free_width_panel(a)
    _results_panel(b)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.9, bottom=0.13)
    out = ROOT / "docs/figures/hero.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


def results():
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    _results_panel(ax, emphasise=False)
    fig.tight_layout()
    out = ROOT / "docs/figures/results_gsm8k.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(hero())
    print(results())

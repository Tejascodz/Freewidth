"""Pareto frontiers, ranking-disagreement metrics and scaling-curve fits."""
from __future__ import annotations

import math


def pareto_front(points: list[tuple[float, float]]) -> list[int]:
    """Indices of points not dominated (lower cost, higher accuracy)."""
    idx = [i for i, (c, a) in enumerate(points) if not (math.isnan(c) or math.isnan(a))]
    idx.sort(key=lambda i: (points[i][0], -points[i][1]))
    front, best = [], -math.inf
    for i in idx:
        if points[i][1] > best + 1e-12:
            front.append(i)
            best = points[i][1]
    return front


def kendall_tau(x: list[float], y: list[float]) -> float:
    pairs = [(a, b) for a, b in zip(x, y) if not (math.isnan(a) or math.isnan(b))]
    n = len(pairs)
    if n < 2:
        return float("nan")
    try:
        from scipy.stats import kendalltau

        return float(kendalltau([p[0] for p in pairs], [p[1] for p in pairs]).statistic)
    except Exception:  # pragma: no cover
        conc = disc = 0
        for i in range(n):
            for j in range(i + 1, n):
                s = (pairs[i][0] - pairs[j][0]) * (pairs[i][1] - pairs[j][1])
                conc += s > 0
                disc += s < 0
        return (conc - disc) / max(1, conc + disc)


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def best_under_budget(rows: list[dict], cost_key: str, budget: float) -> dict | None:
    ok = [r for r in rows if not math.isnan(r[cost_key]) and r[cost_key] <= budget]
    return max(ok, key=lambda r: (r["acc"], -r[cost_key])) if ok else None


def fit_saturating(costs: list[float], accs: list[float]):
    """acc(C) = A - B * C^(-alpha). Returns (A, B, alpha) or None."""
    pts = [(c, a) for c, a in zip(costs, accs) if c > 0 and not math.isnan(c)]
    if len(pts) < 3:
        return None
    try:
        import numpy as np
        from scipy.optimize import curve_fit

        c = np.array([p[0] for p in pts])
        a = np.array([p[1] for p in pts])
        c0 = c.min()

        def f(x, A, B, al):
            return A - B * (x / c0) ** (-al)

        p, _ = curve_fit(f, c, a, p0=[min(1.0, a.max() + 0.05), max(1e-3, a.max() - a.min()), 0.5],
                         bounds=([0, 0, 0], [1.0, 10.0, 5.0]), maxfev=20000)
        return float(p[0]), float(p[1]), float(p[2]), float(c0)
    except Exception:
        return None


COST_LABELS = {
    "flops": "FLOPs per question",
    "kinetic": "eFLOPs per question (multi-tenant, memory-aware)",
    "device_s": "Seconds per question (measured device, single tenant)",
    "device_j": "Joules per question (measured device)",
}


def plot_frontiers(rows: list[dict], cost_keys: list[str], path: str, title: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [k for k in cost_keys if any(not math.isnan(r[k]) for r in rows)]
    if not keys:
        return
    fams = sorted({r["family"] for r in rows})
    cmap = plt.get_cmap("tab20")
    color = {f: cmap(i % 20) for i, f in enumerate(fams)}
    mk = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "p", "d"]
    markers = {f: mk[i % len(mk)] for i, f in enumerate(fams)}
    fig, axes = plt.subplots(1, len(keys), figsize=(5.2 * len(keys), 4.4), squeeze=False)
    for ax, k in zip(axes[0], keys):
        pts = [(r[k], r["acc"]) for r in rows]
        for f in fams:
            sub = [r for r in rows if r["family"] == f and not math.isnan(r[k])]
            ax.scatter([r[k] for r in sub], [r["acc"] for r in sub], s=22, alpha=0.75,
                       color=color[f], marker=markers[f], label=f, linewidths=0)
        fr = pareto_front(pts)
        ax.step([pts[i][0] for i in fr], [pts[i][1] for i in fr], where="post",
                color="black", lw=1.2, alpha=0.8, label="Pareto frontier")
        ax.set_xscale("log")
        ax.set_xlabel(COST_LABELS.get(k, k))
        ax.set_ylabel("accuracy")
        ax.grid(alpha=0.25, which="both")
    axes[0][0].legend(fontsize=7, loc="lower right")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_device_curves(profiles: dict, path: str) -> None:
    """Per-sample cost vs wave width for each model: the 'free width' picture."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not profiles:
        return
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for tag, rows in sorted(profiles.items()):
        Ls = sorted({r["L"] for r in rows})
        L = Ls[-1]
        by_w: dict = {}
        for r in rows:
            if r["L"] == L:
                by_w.setdefault(r["w"], []).append(r["seconds"])
        ws = sorted(by_w)
        base = sum(by_w[ws[0]]) / len(by_w[ws[0]]) / ws[0]
        per = [sum(by_w[w]) / len(by_w[w]) / w / base for w in ws]
        ax.plot(ws, per, marker="o", label=f"{tag} (L={L})")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("wave width (parallel samples)")
    ax.set_ylabel("seconds per sample, relative to width 1")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

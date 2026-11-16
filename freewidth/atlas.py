"""Free-width atlas (experiment E6): measure free width across models, number formats, sequence
lengths and GPUs, and test whether it can be predicted without measuring the target setting.

    python -m freewidth atlas collect --config configs/atlas_t4.yaml --device cuda:0 --shard 0/2
    python -m freewidth atlas analyze runs/atlas_t4 runs/atlas_p100 --out results/atlas

`collect` profiles each setting (one model in one number format on one device) on the same
width x generation-length x prompt-length grid as `calibrate`. It then reads the *measured*
free width directly from the timings at a few operating points (L, P): the largest measured
width whose wave time is within (1 + eps) of a single sample. The value from the fitted cost
curve is recorded alongside it. One JSON line is written per (setting, operating point) to
`atlas.jsonl` as soon as it is measured, so an interrupted run keeps everything it finished,
and a rerun skips finished settings.

`analyze` compares three predictors of log2(free width), none of which may see the setting
being predicted:
- `median`: the median of the other settings (a no-information baseline);
- `roofline`: a weights-only roofline. A decode step reads every weight once (time
  weight_bytes / bandwidth) and does about 2 * params * w FLOPs, so the step stays within
  (1 + eps) of a single sample while w <= (1 + eps) * peak_flops * bytes_per_param /
  (2 * bandwidth). Note that this does not depend on model size;
- `regression`: ridge regression on setting features, evaluated leave-one-model-out.
"""
from __future__ import annotations

import json
import math
import os
import statistics as st
import time
from collections import defaultdict

from .logs import get_logger

log = get_logger("atlas")

# Peak dense fp16 throughput (FLOP/s, tensor cores where present) and memory bandwidth
# (bytes/s), from the vendors' datasheets. A sweep config can override or extend these.
GPU_SPECS: dict[str, tuple[float, float]] = {
    "Tesla T4": (65e12, 320e9),
    "Tesla P100-PCIE-16GB": (18.7e12, 732e9),
    "NVIDIA L4": (121e12, 300e9),
    "NVIDIA A100-SXM4-40GB": (312e12, 1555e9),
    "NVIDIA A100-SXM4-80GB": (312e12, 2039e9),
}

DEFAULTS = dict(eps=0.25, widths=[1, 2, 4, 8, 16, 32, 64], gen_lens=[1, 32, 128],
                prompt_lens=[128, 512], repeats=2, operating_points=[[32, 128], [128, 128], [128, 512]])


# ---------------------------------------------------------------------------------- collect
def load_sweep(path: str) -> dict:
    import yaml

    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    out = {**DEFAULTS, **{k: v for k, v in cfg.items() if k != "settings"}}
    out["settings"] = cfg.get("settings") or []
    names = [s["name"] for s in out["settings"]]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate setting names in {path}")
    for s in out["settings"]:
        missing = {"name", "model_id"} - set(s) if s.get("backend", "hf") == "hf" else {"name"} - set(s)
        if missing:
            raise ValueError(f"setting {s.get('name')!r} is missing {sorted(missing)}")
    return out


def measured_free_width(rows: list[dict], L: int, P: int, eps: float) -> tuple[int | None, bool, dict]:
    """Free width read straight from timings at one (L, P): the largest measured width whose
    mean time is within (1 + eps) of the mean single-sample time. Returns (width, censored,
    ratios); `censored` means even the widest measured wave was within budget, so the true
    free width is at least this value."""
    by_w: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        if r["L"] == L and r["P"] == P:
            by_w[r["w"]].append(r["seconds"])
    if 1 not in by_w:
        return None, False, {}
    base = st.mean(by_w[1])
    ratios = {w: st.mean(v) / base for w, v in sorted(by_w.items())}
    ok = [w for w, x in ratios.items() if x <= 1.0 + eps]
    best = max(ok) if ok else 1
    return best, best == max(ratios), ratios


def _bytes_per_param(setting: dict) -> float:
    if setting.get("load_in_4bit"):
        return 0.5
    return {"float32": 4.0, "bfloat16": 2.0, "float16": 2.0}.get(setting.get("dtype", "float16"), 2.0)


def collect(config: str, device: str, out_dir: str, shard: tuple[int, int] = (0, 1),
            max_hours: float | None = None) -> str:
    """Measure every setting in this shard that is not already in `atlas.jsonl`."""
    from .backends import make_backend
    from .profiler import _free_cuda, profile_backend

    sw = load_sweep(config)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "atlas.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            recs = [json.loads(x) for x in f if x.strip()]
        done = {(r["setting"], r["device_arg"]) for r in recs if "error" not in r}
    deadline = time.time() + max_hours * 3600 if max_hours else None
    i, n = shard
    mine = [s for k, s in enumerate(sw["settings"]) if k % n == i]
    pc = {"widths": sw["widths"], "gen_lens": sw["gen_lens"], "prompt_lens": sw["prompt_lens"],
          "repeats": sw["repeats"], "warmup": True}
    # YAML 1.1 reads "65.0e12" (no exponent sign) as a string, so convert explicitly
    specs = {**GPU_SPECS, **{k: (float(v[0]), float(v[1])) for k, v in (sw.get("gpu_specs") or {}).items()}}
    with open(path, "a") as f:
        for s in mine:
            if (s["name"], device) in done:
                log.info("already measured", extra={"setting": s["name"]})
                continue
            if deadline is not None and time.time() > deadline:
                log.warning("time budget spent; stopping", extra={"setting": s["name"]})
                break
            t0 = time.time()
            spec_in = {k: v for k, v in s.items() if k not in ("name", "arch", "notes")}
            spec_in.setdefault("backend", "hf")
            spec_in.update(tag=s["name"], device=device)
            try:
                b = make_backend(spec_in)
            except Exception as e:  # noqa: BLE001 - record and continue with the next setting
                f.write(json.dumps({"setting": s["name"], "device_arg": device, "error": f"load: {e}"[:500]}) + "\n")
                f.flush()
                log.warning("setting failed to load", extra={"setting": s["name"], "error": str(e)[:200]})
                _free_cuda()
                continue
            try:
                sub = os.path.join(out_dir, f"{s['name']}__{device.replace(':', '')}")
                fit = profile_backend(b, pc, sub, typical_L=128, typical_P=128, eps=sw["eps"], log=log.info)
                with open(os.path.join(sub, "profile.jsonl")) as pf:
                    rows = [json.loads(x) for x in pf if x.strip()]
                gpu = b.device_name()
                flops, bw = specs.get(gpu, (None, None))
                mspec = b.spec()
                weight_bytes = _weight_bytes(b, mspec, s)
                from .costs import DeviceFit

                dfit = DeviceFit.from_dict(fit)
                for L, P in sw["operating_points"]:
                    w, censored, ratios = measured_free_width(rows, L, P, sw["eps"])
                    rec = {
                        "setting": s["name"], "device_arg": device, "gpu": gpu, "model_id": s.get("model_id"),
                        "arch": s.get("arch", "unknown"), "dtype": s.get("dtype", "float16"),
                        "quant": "nf4" if s.get("load_in_4bit") else "none",
                        "bytes_per_param": _bytes_per_param(s), "params": mspec.params if mspec else None,
                        "n_layers": mspec.n_layers if mspec else None,
                        "full_attn_layers": mspec.full_layers if mspec else None,
                        "weight_bytes": weight_bytes, "gpu_flops": flops, "gpu_bw": bw,
                        "L": L, "P": P, "eps": sw["eps"], "free_width": w, "censored": censored,
                        "max_width_measured": max(ratios) if ratios else None,
                        "ratios": {str(k): round(v, 4) for k, v in ratios.items()},
                        "fit_free_width": dfit.free_width(L, P, sw["eps"]), "fit_r2": fit.get("r2"),
                        "step_seconds_w1": _step_seconds(rows, L, P),
                        "minutes": round((time.time() - t0) / 60, 1),
                    }
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                log.info("setting done", extra={"setting": s["name"], "minutes": round((time.time() - t0) / 60, 1)})
            except Exception as e:  # noqa: BLE001 - one bad setting must not end the sweep
                f.write(json.dumps({"setting": s["name"], "device_arg": device, "error": f"profile: {e}"[:500]}) + "\n")
                f.flush()
                log.warning("setting failed", extra={"setting": s["name"], "error": str(e)[:200]})
            finally:
                b.close()
                _free_cuda()
    return path


def _weight_bytes(backend, mspec, setting) -> float | None:
    model = getattr(backend, "model", None)
    if model is not None and hasattr(model, "get_memory_footprint"):
        try:
            return float(model.get_memory_footprint())
        except Exception:  # noqa: BLE001
            pass
    if mspec is not None and mspec.params:
        return mspec.params * _bytes_per_param(setting)
    return None


def _step_seconds(rows, L, P) -> float | None:
    """Per-token decode time of a single sequence: (T(1, L, P) - T(1, 1, P)) / (L - 1)."""
    t = {r["L"]: [] for r in rows}
    for r in rows:
        if r["w"] == 1 and r["P"] == P:
            t[r["L"]].append(r["seconds"])
    if L <= 1 or not t.get(L) or not t.get(1):
        return None
    return max(0.0, (st.mean(t[L]) - st.mean(t[1])) / (L - 1))


# ---------------------------------------------------------------------------------- analyze
def load_atlas(paths: list[str]) -> tuple[list[dict], list[dict]]:
    ok, failed = [], []
    for p in paths:
        fp = os.path.join(p, "atlas.jsonl") if os.path.isdir(p) else p
        with open(fp) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                (failed if "error" in r or not r.get("free_width") else ok).append(r)
    return ok, failed


def roofline_width(r: dict) -> float | None:
    if not r.get("gpu_flops") or not r.get("gpu_bw"):
        return None
    return (1.0 + r["eps"]) * r["gpu_flops"] * r["bytes_per_param"] / (2.0 * r["gpu_bw"])


def features(r: dict) -> list[float]:
    full = r.get("full_attn_layers") or 0
    layers = r.get("n_layers") or 1
    ridge = (r["gpu_flops"] / r["gpu_bw"]) if r.get("gpu_flops") and r.get("gpu_bw") else 1.0
    return [1.0, math.log2(max(r.get("weight_bytes") or 1.0, 1.0)), math.log2(r["L"]), math.log2(r["P"]),
            full / layers, 1.0 if r.get("quant") != "none" else 0.0, math.log2(ridge)]


def _ridge_fit(X, y, lam: float = 1e-2):
    import numpy as np

    X, y = np.asarray(X, float), np.asarray(y, float)
    reg = lam * np.eye(X.shape[1])
    reg[0, 0] = 0.0   # do not shrink the intercept
    return np.linalg.solve(X.T @ X + reg, X.T @ y)


def evaluate(rows: list[dict], group_key: str = "model_id") -> dict:
    """Leave-one-group-out predictions of log2(free width) for each predictor."""
    import numpy as np

    y = [math.log2(r["free_width"]) for r in rows]
    groups = sorted({r.get(group_key) or r["setting"] for r in rows})
    preds = {"median": [None] * len(rows), "roofline": [None] * len(rows), "regression": [None] * len(rows)}
    for g in groups:
        test = [i for i, r in enumerate(rows) if (r.get(group_key) or r["setting"]) == g]
        train = [i for i in range(len(rows)) if i not in test]
        for i in test:
            rw = roofline_width(rows[i])
            preds["roofline"][i] = math.log2(rw) if rw else None
            preds["median"][i] = st.median(y[j] for j in train) if train else None
        if len(train) >= len(features(rows[0])) + 1:
            coef = _ridge_fit([features(rows[j]) for j in train], [y[j] for j in train])
            for i in test:
                preds["regression"][i] = float(np.dot(features(rows[i]), coef))
    out = {}
    for name, p in preds.items():
        pairs = [(y[i], p[i]) for i in range(len(rows)) if p[i] is not None]
        if not pairs:
            out[name] = None
            continue
        err = [abs(a - b) for a, b in pairs]
        out[name] = {"n": len(pairs), "mae_log2": round(st.mean(err), 3),
                     "within_2x": round(sum(e <= 1.0 for e in err) / len(err), 3),
                     "median_ratio_pred_over_measured": round(2 ** st.median(b - a for a, b in pairs), 2)}
    out["_predictions"] = preds
    return out


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    from scipy.stats import spearmanr

    return float(spearmanr(xs, ys).statistic)


def size_trend(rows: list[dict]) -> list[dict]:
    """H12: Spearman correlation of weight bytes vs free width within each (GPU, arch, quant,
    L, P) group with at least 3 models."""
    g = defaultdict(list)
    for r in rows:
        g[(r["gpu"], r["arch"], r["quant"], r["L"], r["P"])].append(r)
    out = []
    for k, rs in sorted(g.items()):
        rho = spearman([r["weight_bytes"] or 0 for r in rs], [r["free_width"] for r in rs])
        if rho is not None:
            out.append({"gpu": k[0], "arch": k[1], "quant": k[2], "L": k[3], "P": k[4],
                        "n_models": len(rs), "spearman": round(rho, 3)})
    return out


def analyze(paths: list[str], out_dir: str) -> str:
    rows, failed = load_atlas(paths)
    if not rows:
        raise ValueError("no successful atlas rows")
    os.makedirs(out_dir, exist_ok=True)
    ev_model = evaluate(rows, "model_id")
    ev_gpu = evaluate(rows, "gpu") if len({r["gpu"] for r in rows}) > 1 else None
    trend = size_trend(rows)
    roof = [(r, roofline_width(r)) for r in rows]
    roof_ratio = [rw / r["free_width"] for r, rw in roof if rw]
    L = ["# Free-width atlas (E6)", "",
         f"{len({r['setting'] for r in rows})} settings, {len({r['gpu'] for r in rows})} GPU type(s), "
         f"{len(rows)} (setting, operating point) measurements; {len(failed)} failed or empty.", "",
         "## Measurements", "",
         "| setting | GPU | arch | format | weights (GB) | L | P | free width | censored | fitted | roofline |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r, rw in sorted(roof, key=lambda t: (t[0]["gpu"], t[0]["setting"], t[0]["L"], t[0]["P"])):
        wb = f"{r['weight_bytes'] / 1e9:.2f}" if r.get("weight_bytes") else "—"
        rws = f"{rw:.0f}" if rw else "—"
        fmt = r["quant"] if r["quant"] != "none" else r["dtype"]
        L.append(f"| {r['setting']} | {r['gpu']} | {r['arch']} | {fmt} | {wb} | {r['L']} | {r['P']} "
                 f"| {r['free_width']} | {'yes' if r['censored'] else 'no'} | {r['fit_free_width']} | {rws} |")
    L += ["", "## H11: does a weights-only roofline predict free width?", ""]
    if roof_ratio:
        L.append(f"Roofline / measured free width: median {st.median(roof_ratio):.1f}x, "
                 f"min {min(roof_ratio):.1f}x, max {max(roof_ratio):.1f}x over {len(roof_ratio)} measurements.")
    L += ["", "## H12: does free width fall with model size?", "",
          "| GPU | arch | format | L | P | models | Spearman (weights vs free width) |", "|---|---|---|---|---|---|---|"]
    for t in trend:
        L.append(f"| {t['gpu']} | {t['arch']} | {t['quant']} | {t['L']} | {t['P']} | {t['n_models']} | {t['spearman']} |")
    L += ["", "## H13: can free width be predicted for an unseen model?", "",
          "Error in log2 units (1.0 = off by a factor of 2), leave-one-model-out.", "",
          "| predictor | n | MAE (log2) | within 2x | median predicted / measured |", "|---|---|---|---|---|"]
    for name in ("median", "roofline", "regression"):
        m = ev_model.get(name)
        if m:
            L.append(f"| {name} | {m['n']} | {m['mae_log2']} | {m['within_2x']:.0%} | {m['median_ratio_pred_over_measured']} |")
    if ev_gpu:
        L += ["", "Leave-one-GPU-out (exploratory):", "", "| predictor | n | MAE (log2) | within 2x |", "|---|---|---|---|"]
        for name in ("median", "roofline", "regression"):
            m = ev_gpu.get(name)
            if m:
                L.append(f"| {name} | {m['n']} | {m['mae_log2']} | {m['within_2x']:.0%} |")
    if failed:
        L += ["", "## Failed settings", ""] + [f"- {r['setting']} ({r.get('device_arg')}): {r.get('error', 'no free width')}"
                                                for r in failed]
    md = "\n".join(L) + "\n"
    with open(os.path.join(out_dir, "ATLAS.md"), "w") as f:
        f.write(md)
    summary = {"by_model": {k: v for k, v in ev_model.items() if not k.startswith("_")},
               "by_gpu": None if ev_gpu is None else {k: v for k, v in ev_gpu.items() if not k.startswith("_")},
               "size_trend": trend, "roofline_over_measured": roof_ratio, "n_failed": len(failed)}
    with open(os.path.join(out_dir, "atlas_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    _plot(rows, os.path.join(out_dir, "atlas.png"))
    return os.path.join(out_dir, "ATLAS.md")


def _plot(rows, path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    pts = [r for r in rows if r["L"] == 128 and r["P"] == 128 and r.get("weight_bytes")]
    if not pts:
        return
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#e34948"]
    groups = sorted({(r["gpu"], r["quant"] != "none") for r in pts})
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    for k, (gpu, q) in enumerate(groups):
        rs = sorted([r for r in pts if r["gpu"] == gpu and (r["quant"] != "none") == q], key=lambda r: r["weight_bytes"])
        col = palette[k % len(palette)]
        lab = f"{gpu}, {'4-bit' if q else 'fp16'}"
        ax.plot([r["weight_bytes"] / 1e9 for r in rs], [r["free_width"] for r in rs], "o-", color=col, lw=2, ms=7,
                label=lab, markeredgecolor="white", markeredgewidth=1.5)
        rw = roofline_width(rs[0])
        if rw:
            ax.axhline(rw, color=col, ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log", base=2)
    ax.set_xlabel("weights in GPU memory (GB)")
    ax.set_ylabel("measured free width (L=128, P=128)")
    ax.set_title("Free width vs model size (dashed: weights-only roofline)", fontsize=11, loc="left")
    ax.grid(color="#e5e5e5", lw=0.8)
    ax.legend(frameon=False, fontsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

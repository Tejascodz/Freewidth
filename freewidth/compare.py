"""Compare finished runs (e.g. hybrid vs dense family, or T4 vs Mac) side by side.

H2 check: under the memory-aware (kinetic) cost, how much more does wide sampling on the
smallest model pay than one sample from the largest model? Kinetics predicts a large
penalty for dense models; hybrids keep a KV cache in only some layers, so the penalty
should shrink.
"""
from __future__ import annotations

import csv
import json
import math
import os


def _load(run: str):
    a = os.path.join(run, "analysis")
    with open(os.path.join(a, "results.csv")) as f:
        rows = [{k: (float(v) if k not in ("strategy", "family", "models") else v) for k, v in r.items()}
                for r in csv.DictReader(f)]
    with open(os.path.join(a, "disagreement.json")) as f:
        dis = json.load(f)
    specs = {}
    for tag in sorted(os.listdir(os.path.join(run, "models"))):
        for fn in ("spec.json", "meta.json"):
            p = os.path.join(run, "models", tag, fn)
            if os.path.exists(p):
                with open(p) as f:
                    j = json.load(f)
                j = j.get("spec", j) if fn == "meta.json" else j
                if j and "params" in j:
                    specs[tag] = j
                    break
    return rows, dis, specs


def _row(rows, family, models, **kw):
    for r in rows:
        if r["family"] == family and r["models"] == models and all(f"{k}={v}" in r["strategy"] for k, v in kw.items()):
            return r
    return None


def summarize(run: str) -> dict:
    rows, dis, specs = _load(run)
    tags = sorted(specs, key=lambda t: specs[t]["params"])
    out = {"run": os.path.basename(os.path.abspath(run)), "models": tags}
    if tags:
        small, big = tags[0], tags[-1]
        sp = specs[small]
        nl, nf = sp.get("n_layers") or 0, sp.get("n_full_attn_layers")
        out["hybrid"] = bool(nf is not None and nl and nf < nl)
        wide = None
        for n in (16, 8, 32, 4):
            wide = _row(rows, "sc", small, n=n)
            if wide:
                break
        one = _row(rows, "single", big)
        if wide and one and wide["flops"] > 0 and one["flops"] > 0:
            pen_small = wide["kinetic"] / wide["flops"]
            pen_big = one["kinetic"] / one["flops"]
            out["kv_penalty_small_wide"] = pen_small
            out["kv_penalty_big_single"] = pen_big
            out["kv_penalty_ratio"] = pen_small / pen_big
            out["wide_strategy"] = wide["strategy"]
    kfront = dis.get("fronts", {}).get("kinetic", [])
    if kfront and tags:
        small_only = [s for s in kfront if all(t == tags[0] for t in _models_of(s, rows))]
        out["kinetic_front_small_share"] = len(small_only) / len(kfront)
    out["kendall_flops_device"] = dis.get("kendall_tau", {}).get("flops~device_s", math.nan)
    out["jaccard_flops_device"] = dis.get("pareto_jaccard", {}).get("flops~device_s", math.nan)
    ov = [o for o in dis.get("overruns", []) if o["guide"] == "flops" and o["deploy"] == "device_s"]
    out["overrun_flops_to_device_max"] = ov[0]["max"] if ov else math.nan
    return out


def _models_of(strategy: str, rows) -> list[str]:
    for r in rows:
        if r["strategy"] == strategy:
            return r["models"].split("+")
    return []


def run(runs: list[str], out_path: str | None = None) -> str:
    sums = [summarize(r) for r in runs]
    cols = [("run", "run"), ("hybrid", "hybrid"), ("kv_penalty_ratio", "KV penalty ratio (small wide / big single)"),
            ("kinetic_front_small_share", "small-only share of kinetic frontier"),
            ("kendall_flops_device", "Kendall tau FLOPs~device"), ("jaccard_flops_device", "Pareto Jaccard FLOPs~device"),
            ("overrun_flops_to_device_max", "worst overrun FLOPs->device")]
    lines = ["# Cross-run comparison", "", "| " + " | ".join(c[1] for c in cols) + " |",
             "|" + "---|" * len(cols)]
    for s in sums:
        cells = []
        for k, _ in cols:
            v = s.get(k, math.nan)
            cells.append(f"{v:.3f}" if isinstance(v, float) and not math.isnan(v) else
                         ("n/a" if isinstance(v, float) else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "H2 predicts a lower KV penalty ratio and a higher small-only kinetic-frontier share "
              "for hybrid runs than for dense runs at matched sizes."]
    text = "\n".join(lines) + "\n"
    if out_path:
        with open(out_path, "w") as f:
            f.write(text)
    return text

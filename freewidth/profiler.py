"""Measure the device's wave cost curve T(width, gen_len, prompt_len) and fit it."""
from __future__ import annotations

import json
import os
import time

from .backends import make_backend
from .config import model_dir
from .costs import fit_device_curve
from .logs import get_logger

log = get_logger("profiler")


def profile_model(cfg: dict, mcfg: dict, log=print) -> dict:
    tag = mcfg["tag"]
    pc = cfg["profile"]
    out_dir = model_dir(cfg, tag)
    src = mcfg.get("profile_from")
    if src:  # same weights on the same device (e.g. a thinking-budget variant): reuse the fit
        import shutil

        for fn in ("device_fit.json", "spec.json", "profile.jsonl"):
            p = os.path.join(model_dir(cfg, src), fn)
            if os.path.exists(p):
                shutil.copy(p, os.path.join(out_dir, fn))
        log(f"[{tag}] reused device profile of {src}")
        with open(os.path.join(out_dir, "device_fit.json")) as f:
            return json.load(f)
    backend = make_backend(mcfg)
    if not backend.measures_device:
        log(f"[{tag}] backend does not run on this device; skipping profile")
        backend.close()
        return {}
    d = profile_backend(backend, pc, out_dir,
                        typical_L=int(cfg["sampling"]["max_new_tokens"] * 0.4),
                        typical_P=int(pc.get("typical_prompt_len", pc["prompt_len"])),
                        eps=cfg["analysis"]["free_width_eps"], log=log)
    backend.close()
    return d


def profile_backend(backend, pc: dict, out_dir: str, *, typical_L: int, typical_P: int,
                    eps: float, log=print) -> dict:
    """Profile an already-loaded backend, fit T(w, L, P), write device_fit.json + spec.json."""
    tag = backend.tag
    os.makedirs(out_dir, exist_ok=True)
    device = backend.device_name()
    log(f"[{tag}] profiling on {device}")
    prompt_lens = pc.get("prompt_lens") or [pc.get("prompt_len", 160)]
    backend.timed_decode(1, 8, prompt_lens[0])  # warm-up (CUDA context, kernels)
    rows = []
    # Two or more prompt lengths are needed to separate the per-row cost (b) from the
    # context-dependent attention/KV cost (g) in the fit.
    if pc.get("warmup", True):
        # JIT kernels (Triton, e.g. flash-linear-attention) compile and autotune per shape on
        # first use; one untimed call per (width, prompt length) keeps that out of the fit.
        for Pp in prompt_lens:
            for w in pc["widths"]:
                try:
                    backend.timed_decode(w, 2, Pp)
                except RuntimeError:
                    _free_cuda()
                    break
    for Pp in prompt_lens:
        for L in pc["gen_lens"]:
            for w in pc["widths"]:
                failed = False
                for r in range(pc.get("repeats", 2)):
                    try:
                        s, j = backend.timed_decode(w, L, Pp)
                    except RuntimeError as e:  # OOM at large widths: stop widening
                        log(f"[{tag}] w={w} L={L} P={Pp} failed ({str(e)[:80]}); stopping widths here")
                        _free_cuda()
                        failed = True
                        break
                    rows.append({"w": w, "L": L, "P": Pp, "seconds": s, "joules": j, "rep": r})
                    log(f"[{tag}] P={Pp:>4} w={w:>3} L={L:>4} -> {s:7.3f}s" + (f" {j:8.1f}J" if j else ""))
                if failed:
                    break
    if len(rows) < 5:
        raise RuntimeError(f"[{tag}] too few profile points ({len(rows)}) to fit a cost curve")
    with open(os.path.join(out_dir, "profile.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    fit = fit_device_curve(rows, tag, device)
    wf = fit.free_width(typical_L, typical_P, eps)
    d = fit.to_dict()
    d["free_width"] = wf
    d["typical_gen_len"] = typical_L
    d["typical_prompt_len"] = typical_P
    d["eps"] = eps
    d["profiled_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(out_dir, "device_fit.json"), "w") as f:
        json.dump(d, f, indent=2)
    log(f"[{tag}] fit R^2={fit.r2:.4f}; free width (eps={eps}) at L={typical_L}: {wf}")
    _write_spec(backend, out_dir)
    return d


def _write_spec(backend, out_dir):
    spec = backend.spec()
    if spec is not None:
        with open(os.path.join(out_dir, "spec.json"), "w") as f:
            json.dump(spec.to_dict(), f, indent=2)


def _free_cuda():
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception as e:  # noqa: BLE001 - best-effort cleanup
        log.debug("cuda cache not freed", extra={"error": str(e)})


def run(cfg: dict, only: list[str] | None = None, log=print) -> None:
    for m in cfg["models"]:
        if only and m["tag"] not in only:
            continue
        profile_model(cfg, m, log)

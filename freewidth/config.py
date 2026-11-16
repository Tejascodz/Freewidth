from __future__ import annotations

import copy
import os

import yaml

DEFAULTS = {
    "output_dir": "runs/default",
    "dataset": {"name": "gsm8k", "split": "test", "limit": 300, "seed": 0},
    "prompt": {},
    "sampling": {"temperature": 0.7, "top_p": 1.0, "max_new_tokens": 768},
    "bank": {"samples_per_question": 32, "wave_width": 16, "seed": 1234},
    "verify": {"pairs": [], "batch": 8, "max_chars": 6000},
    "profile": {"widths": [1, 2, 4, 8, 16, 32, 64], "gen_lens": [1, 64, 256],
                "prompt_len": 160, "prompt_lens": [128, 1024], "repeats": 2},
    "analysis": {
        "trials": 20,
        "bootstrap": 500,
        "kinetic_intensity": 295.0,   # H100 dense BF16 FLOP/byte, the multi-tenant reference
        "free_width_eps": 0.25,
        "static_batch": True,
        "budget_quantiles": [0.1, 0.25, 0.5, 0.75, 0.9],
    },
    "models": [],
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str, validate: bool = True) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    cfg = _merge(DEFAULTS, cfg)
    cfg["_path"] = os.path.abspath(path)
    if validate:
        from .schema import validate_research_config

        validate_research_config(cfg)
    return cfg


def model_dir(cfg: dict, tag: str) -> str:
    d = os.path.join(cfg["output_dir"], "models", tag)
    os.makedirs(d, exist_ok=True)
    return d


def public_model_cfg(m: dict) -> dict:
    """Model config without anything that looks like a secret."""
    return {k: v for k, v in m.items() if "key" not in k.lower() or k == "api_key_env"}

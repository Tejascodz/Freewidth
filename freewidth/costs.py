"""Three ways to price the same inference trace.

1. `flops`   - the classic test-time-scaling accounting (Snell et al. 2024,
               Wu et al. 2024): 2 * params * tokens + attention FLOPs.
2. `kinetic` - multi-tenant datacenter pricing in the spirit of Kinetics
               (Sadhukhan et al. 2025): compute + KV/state memory traffic, with
               weight reads assumed amortised across many concurrent users.
               Memory bytes are converted to FLOP-equivalents with the
               reference accelerator's arithmetic intensity.
3. `device`  - single-tenant, measured on the actual device. Each *wave* of
               parallel samples is priced with a cost curve fitted from
               microbenchmarks (see profiler.py). Weight reads are paid once
               per decode step and shared by every sequence in the wave.

The hypothesis under test is that (1)/(2) and (3) rank strategies differently.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class ModelSpec:
    name: str
    params: float                 # total parameters
    active_params: float | None = None   # for MoE; defaults to params
    n_layers: int = 0
    n_full_attn_layers: int | None = None  # hybrid models: only these keep a KV cache
    n_heads: int = 0
    n_kv_heads: int = 0
    head_dim: int = 0
    kv_dtype_bytes: int = 2
    linear_state_bytes: float = 0.0  # per-sequence recurrent state read per token (hybrid layers)

    @property
    def p_active(self) -> float:
        return self.active_params or self.params

    @property
    def full_layers(self) -> int:
        return self.n_layers if self.n_full_attn_layers is None else self.n_full_attn_layers

    @property
    def kv_bytes_per_token(self) -> float:
        return 2.0 * self.full_layers * self.n_kv_heads * self.head_dim * self.kv_dtype_bytes

    def attn_flops_per_token(self, ctx: float) -> float:
        # QK^T and AV for one new token against `ctx` cached positions.
        return 4.0 * self.full_layers * self.n_heads * self.head_dim * ctx

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ModelSpec:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_hf_config(cls, name: str, cfg: dict, params: float) -> ModelSpec:
        """Build a spec from a HF `config.to_dict()`; handles nested text configs."""
        c = cfg.get("text_config") or cfg
        n_layers = int(c.get("num_hidden_layers", 0))
        n_heads = int(c.get("num_attention_heads", 0))
        n_kv = int(c.get("num_key_value_heads", n_heads) or n_heads)
        hidden = int(c.get("hidden_size", 0))
        head_dim = int(c.get("head_dim") or (hidden // n_heads if n_heads else 0))
        full = None
        layer_types = c.get("layer_types")
        if isinstance(layer_types, list) and layer_types:
            full = sum(1 for t in layer_types if "full" in str(t) or t == "attention")
        elif c.get("full_attention_interval"):
            full = n_layers // int(c["full_attention_interval"])
        lin_state = 0.0
        if full is not None and full < n_layers:
            # Gated-DeltaNet style state: heads * k_dim * v_dim per linear layer.
            lh = int(c.get("linear_num_value_heads", c.get("linear_num_key_heads", n_heads)) or n_heads)
            kd = int(c.get("linear_key_head_dim", head_dim) or head_dim)
            vd = int(c.get("linear_value_head_dim", head_dim) or head_dim)
            lin_state = float((n_layers - full) * lh * kd * vd * 4)  # fp32 state
        return cls(
            name=name, params=float(params), n_layers=n_layers, n_full_attn_layers=full,
            n_heads=n_heads, n_kv_heads=n_kv, head_dim=head_dim, linear_state_bytes=lin_state,
        )


# --------------------------------------------------------------------------- #
# Per-sample costs (flops, kinetic)
# --------------------------------------------------------------------------- #

def _attn_sum(spec: ModelSpec, prompt: int, gen: int) -> float:
    # sum_{t=1..gen} attn_flops(prompt + t)  (closed form)
    s = gen * prompt + gen * (gen + 1) / 2.0
    return 4.0 * spec.full_layers * spec.n_heads * spec.head_dim * s


def flops_sample(spec: ModelSpec, prompt: int, gen: int) -> float:
    return 2.0 * spec.p_active * gen + _attn_sum(spec, prompt, gen)


def flops_prefill(spec: ModelSpec, prompt: int) -> float:
    return 2.0 * spec.p_active * prompt + _attn_sum(spec, 0, prompt)


def kinetic_sample(spec: ModelSpec, prompt: int, gen: int, intensity: float) -> float:
    compute = flops_sample(spec, prompt, gen)
    ctx_sum = gen * prompt + gen * (gen + 1) / 2.0
    mem = spec.kv_bytes_per_token * ctx_sum + spec.linear_state_bytes * gen
    return compute + intensity * mem


def kinetic_prefill(spec: ModelSpec, prompt: int, intensity: float) -> float:
    return flops_prefill(spec, prompt) + intensity * spec.kv_bytes_per_token * prompt


# --------------------------------------------------------------------------- #
# Device cost curve (single tenant)
# --------------------------------------------------------------------------- #

@dataclass
class DeviceFit:
    """T(w, L, P) = e + d*w*P + a*L + b*w*L + g*w*(L*P + L(L+1)/2)

    a  ~ per-step fixed cost (weight reads, kernel launches)
    b  ~ per-step per-sequence cost (MLP compute, sampling)
    g  ~ per-step per-sequence per-context-token cost (KV reads / attention)
    d  ~ prefill cost per prompt token per sequence
    e  ~ constant overhead per call
    """

    model: str
    coef: dict = field(default_factory=dict)          # keys: e, d, a, b, g
    energy_coef: dict | None = None
    r2: float | None = None
    energy_r2: float | None = None
    device: str = ""

    @staticmethod
    def features(w: float, L: float, P: float) -> list[float]:
        return [1.0, w * P, L, w * L, w * (L * P + L * (L + 1) / 2.0)]

    def _eval(self, coef: dict, w: float, L: float, P: float) -> float:
        f = self.features(w, L, P)
        c = [coef["e"], coef["d"], coef["a"], coef["b"], coef["g"]]
        return max(0.0, sum(x * y for x, y in zip(f, c)))

    def seconds(self, w: float, L: float, P: float) -> float:
        return self._eval(self.coef, w, L, P)

    def joules(self, w: float, L: float, P: float) -> float | None:
        if not self.energy_coef:
            return None
        return self._eval(self.energy_coef, w, L, P)

    def free_width(self, L: float, P: float, eps: float = 0.25,
                   candidates=(1, 2, 4, 8, 16, 32, 64, 128)) -> int:
        """Widest wave whose wall-clock is within (1+eps) of a single sample."""
        base = self.seconds(1, L, P)
        best = 1
        for w in candidates:
            if self.seconds(w, L, P) <= (1.0 + eps) * base:
                best = w
        return best

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> DeviceFit:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def fit_device_curve(rows: list[dict], model: str, device: str = "") -> DeviceFit:
    """Non-negative least squares over profiler rows {w, L, P, seconds, joules}."""
    import numpy as np

    X = np.array([DeviceFit.features(r["w"], r["L"], r["P"]) for r in rows], dtype=float)
    keys = ["e", "d", "a", "b", "g"]

    def _fit(y):
        y = np.asarray(y, dtype=float)
        scale = X.max(axis=0)
        scale[scale == 0] = 1.0
        Xs = X / scale
        try:
            from scipy.optimize import nnls

            coef, _ = nnls(Xs, y)
        except Exception:  # pragma: no cover - scipy missing
            coef, *_ = np.linalg.lstsq(Xs, y, rcond=None)
            coef = np.clip(coef, 0, None)
        coef = coef / scale
        pred = X @ coef
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum()) or 1e-12
        return dict(zip(keys, map(float, coef))), 1.0 - ss_res / ss_tot

    coef, r2 = _fit([r["seconds"] for r in rows])
    ecoef, er2 = None, None
    if rows and all(r.get("joules") is not None for r in rows):
        ecoef, er2 = _fit([r["joules"] for r in rows])
    return DeviceFit(model=model, coef=coef, energy_coef=ecoef, r2=r2, energy_r2=er2, device=device)


def wave_device_cost(fit: DeviceFit, width: int, gen_lens: list[int], prompt: int,
                     static_batch: bool = True) -> tuple[float, float | None]:
    """Seconds (and joules) for one wave of `width` parallel samples.

    static_batch=True  -> every row runs until the longest one ends (HF generate).
    static_batch=False -> continuous batching: active width shrinks as rows finish.
    """
    if not gen_lens:
        return 0.0, 0.0 if fit.energy_coef else None
    if static_batch:
        L = max(gen_lens)
        return fit.seconds(width, L, prompt), fit.joules(width, L, prompt)
    # continuous batching: integrate over segments where the active width is constant
    lens = sorted(gen_lens)
    secs, joules = fit.seconds(width, 0, prompt), fit.joules(width, 0, prompt)  # prefill
    prev = 0
    for i, L in enumerate(lens):
        active = len(lens) - i
        if prev < L:
            s1, s0 = fit.seconds(active, L, prompt), fit.seconds(active, prev, prompt)
            secs += s1 - s0
            if joules is not None:
                joules += (fit.joules(active, L, prompt) or 0) - (fit.joules(active, prev, prompt) or 0)
        prev = L
    return secs, joules

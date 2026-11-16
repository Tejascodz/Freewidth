"""Load a sample bank and price replayed strategy traces under every cost model."""
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field

from .costs import DeviceFit, ModelSpec, flops_prefill, flops_sample, kinetic_prefill, kinetic_sample, wave_device_cost
from .grading import answer_key


@dataclass(frozen=True)
class Sample:
    key: str | None
    correct: bool
    gen: int
    sc: float | None
    lp: float | None
    sid: str = ""          # "<wave>-<i>", stable id inside one (model, question)


@dataclass
class ModelBank:
    tag: str
    spec: ModelSpec | None
    fit: DeviceFit | None
    free_width: int | None
    samples: dict = field(default_factory=dict)      # qid -> list[Sample]
    prompt_tokens: dict = field(default_factory=dict)  # qid -> int
    measured_waves: list = field(default_factory=list)
    budget: int = 0                                   # max_new_tokens used for this bank
    # verification scores this model assigned to another model's samples:
    # verify[small_tag][qid][sid] = (p_yes, verify_prompt_tokens)
    verify: dict = field(default_factory=dict)

    @property
    def size(self) -> float:
        return self.spec.p_active if self.spec else float("nan")


@dataclass
class Bank:
    models: dict          # tag -> ModelBank
    qids: list
    meta: dict = field(default_factory=dict)

    def tags_by_size(self) -> list[str]:
        """Ascending by parameters, then by generation budget (think-longer variants)."""
        return sorted(self.models, key=lambda t: (math.isnan(self.models[t].size),
                                                  self.models[t].size, self.models[t].budget, t))


def _read_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def load_bank(run_dir: str, cfg: dict | None = None, min_samples: int | None = None) -> Bank:
    mroot = os.path.join(run_dir, "models")
    cfg_models = {m["tag"]: m for m in (cfg or {}).get("models", [])}
    models = {}
    for tag in sorted(os.listdir(mroot)) if os.path.isdir(mroot) else []:
        d = os.path.join(mroot, tag)
        rows = _read_jsonl(os.path.join(d, "samples.jsonl"))
        if not rows:
            continue
        spec = None
        for p in (os.path.join(d, "spec.json"), os.path.join(d, "meta.json")):
            if os.path.exists(p):
                with open(p) as f:
                    j = json.load(f)
                j = j.get("spec", j) if p.endswith("meta.json") else j
                if j and "params" in j:
                    spec = ModelSpec.from_dict(j)
                    break
        if spec is None and cfg_models.get(tag, {}).get("spec"):
            spec = ModelSpec.from_dict({"name": tag, **cfg_models[tag]["spec"]})
        fit, wf = None, None
        fp = os.path.join(d, "device_fit.json")
        if os.path.exists(fp):
            with open(fp) as f:
                j = json.load(f)
            fit, wf = DeviceFit.from_dict(j), j.get("free_width")
        mb = ModelBank(tag, spec, fit, wf)
        mp = os.path.join(d, "meta.json")
        if os.path.exists(mp):
            with open(mp) as f:
                meta = json.load(f)
            mb.budget = int((meta.get("sampling") or {}).get("max_new_tokens") or 0)
        rows.sort(key=lambda r: (r["qid"], r["wave"], r["i"]))
        for r in rows:
            mb.samples.setdefault(r["qid"], []).append(
                Sample(answer_key(r.get("answer")), bool(r["correct"]), int(r["gen_tokens"]),
                       r.get("self_certainty"), r.get("mean_logprob"), f"{r['wave']}-{r['i']}"))
            mb.prompt_tokens[r["qid"]] = int(r.get("prompt_tokens") or 0)
        mb.measured_waves = _read_jsonl(os.path.join(d, "waves.jsonl"))
        for fn in os.listdir(d):
            if fn.startswith("verify_") and fn.endswith(".jsonl"):
                small = fn[len("verify_"):-len(".jsonl")]
                for r in _read_jsonl(os.path.join(d, fn)):
                    mb.verify.setdefault(small, {}).setdefault(r["qid"], {})[r["sid"]] = (
                        float(r["p_yes"]), int(r.get("prompt_tokens") or 0))
        models[tag] = mb
    if not models:
        raise FileNotFoundError(f"no sample banks under {mroot}")
    common = set.intersection(*(set(m.samples) for m in models.values()))
    if min_samples:
        common = {q for q in common if all(len(m.samples[q]) >= min_samples for m in models.values())}
    return Bank(models, sorted(common))


class Stream:
    """Per-trial random order over each (model, question) sample list; draws without replacement."""

    def __init__(self, bank: Bank, trial: int):
        self.bank = bank
        self.trial = trial
        self._pos: dict = {}
        self._perm: dict = {}

    def rng(self, *salt) -> random.Random:
        return random.Random("|".join(map(str, (self.trial,) + salt)))

    def _order(self, tag, qid):
        k = (tag, qid)
        if k not in self._perm:
            n = len(self.bank.models[tag].samples[qid])
            idx = list(range(n))
            random.Random(f"{self.trial}|{tag}|{qid}").shuffle(idx)
            self._perm[k] = idx
            self._pos[k] = 0
        return k

    def take(self, tag: str, qid: str, n: int) -> list[Sample]:
        k = self._order(tag, qid)
        p = self._pos[k]
        idx = self._perm[k][p : p + n]
        self._pos[k] = p + len(idx)
        s = self.bank.models[tag].samples[qid]
        return [s[i] for i in idx]

    def remaining(self, tag: str, qid: str) -> int:
        k = self._order(tag, qid)
        return len(self._perm[k]) - self._pos[k]


@dataclass
class Trace:
    """What a strategy did for one question: an ordered list of waves."""

    key: str | None
    waves: list = field(default_factory=list)   # list[(tag, list[Sample])]
    # prefill-only passes (verification): list[(tag, width, prompt_tokens_per_row)]
    prefills: list = field(default_factory=list)

    def add(self, tag: str, samples: list[Sample]) -> None:
        if samples:
            self.waves.append((tag, samples))

    def add_prefill(self, tag: str, width: int, prompt_tokens: int) -> None:
        if width > 0:
            self.prefills.append((tag, width, prompt_tokens))


def is_key_correct(bank: Bank, qid: str, key: str | None) -> bool:
    if key is None:
        return False
    for mb in bank.models.values():
        for s in mb.samples.get(qid, []):
            if s.key == key:
                return s.correct
    return False


def price_trace(bank: Bank, qid: str, tr: Trace, intensity: float, static_batch: bool = True) -> dict:
    flops = kin = secs = 0.0
    joules: float | None = 0.0
    tokens = 0
    used_models = set()
    nan = float("nan")
    for tag, samples in tr.waves:
        mb = bank.models[tag]
        P = mb.prompt_tokens.get(qid, 0)
        gens = [s.gen for s in samples]
        tokens += sum(gens)
        if mb.spec is not None:
            if tag not in used_models:  # prompt prefilled once per model (prefix cache)
                flops += flops_prefill(mb.spec, P)
                kin += kinetic_prefill(mb.spec, P, intensity)
            flops += sum(flops_sample(mb.spec, P, g) for g in gens)
            kin += sum(kinetic_sample(mb.spec, P, g, intensity) for g in gens)
        else:
            flops = kin = nan
        if mb.fit is not None:
            s, j = wave_device_cost(mb.fit, len(samples), gens, P, static_batch)
            secs += s
            joules = None if (joules is None or j is None) else joules + j
        else:
            secs, joules = nan, None
        used_models.add(tag)
    for tag, width, Pv in tr.prefills:
        # one forward pass over `width` rows of Pv tokens, plus one decoded token per row
        mb = bank.models[tag]
        if mb.spec is not None:
            flops += width * (flops_prefill(mb.spec, Pv) + flops_sample(mb.spec, Pv, 1))
            kin += width * (kinetic_prefill(mb.spec, Pv, intensity) + kinetic_sample(mb.spec, Pv, 1, intensity))
        else:
            flops = kin = nan
        if mb.fit is not None:
            secs += mb.fit.seconds(width, 1, Pv)
            j = mb.fit.joules(width, 1, Pv)
            joules = None if (joules is None or j is None) else joules + j
        else:
            secs, joules = nan, None
    return {"flops": flops, "kinetic": kin, "device_s": secs,
            "device_j": joules if joules is not None else nan,
            "tokens": tokens, "rounds": len(tr.waves) + len(tr.prefills),
            "samples": sum(len(s) for _, s in tr.waves)}

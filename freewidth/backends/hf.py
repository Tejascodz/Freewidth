"""Local Hugging Face transformers backend (CUDA, Apple MPS or CPU).

One `generate` call = one wave: the prompt is repeated `n` times in a single
static batch, so wall-clock and energy are those of a real parallel wave.
"""
from __future__ import annotations

import math
import time

from ..costs import ModelSpec
from ..hw import EnergyMeter, cuda_index
from ..logs import get_logger
from .base import Backend, GenResult

log = get_logger("hf")


class _StatsRecorder:
    """Logits processor that records per-step entropy and self-certainty.

    Self-certainty (Kang et al., 2025) = KL(Uniform || p) averaged over tokens.
    Computed on the distribution the sampler sees; masked (-inf) entries are
    excluded so top-k/top-p truncation does not produce NaNs.
    """

    def __init__(self):
        self.ent: list = []
        self.sc: list = []

    def __call__(self, input_ids, scores):
        import torch

        s = scores.float()
        finite = torch.isfinite(s)
        s = torch.where(finite, s, torch.full_like(s, -1e9))
        logp = torch.log_softmax(s, dim=-1)
        p = logp.exp()
        ent = -(p * torch.where(finite, logp, torch.zeros_like(logp))).sum(-1)
        k = finite.sum(-1).clamp(min=1).float()
        mean_logp = torch.where(finite, logp, torch.zeros_like(logp)).sum(-1) / k
        sc = -torch.log(k) - mean_logp
        self.ent.append(ent.cpu())
        self.sc.append(sc.cpu())
        return scores


class HFBackend(Backend):
    measures_device = True

    def __init__(self, tag: str, model_id: str, device: str = "cuda:0",
                 dtype: str = "float16", load_in_4bit: bool = False,
                 enable_thinking: bool | None = False, trust_remote_code: bool = False,
                 attn_implementation: str | None = None, answer_tokens: int = 64,
                 revision: str | None = None, **_):
        import torch
        from transformers import AutoTokenizer

        self.tag = tag
        self.model_id = model_id
        self.device = device
        self.enable_thinking = enable_thinking
        self.answer_tokens = answer_tokens
        # trust_remote_code runs Python shipped inside the model repo; it is off unless a config
        # turns it on for a model that needs it. `revision` pins the exact Hub commit.
        if revision is None:
            log.warning("model not pinned to a Hub revision; set `revision` for reproducible, "
                        "tamper-evident loads", extra={"model_id": model_id})
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code,
                                                 revision=revision)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        torch_dtype = getattr(torch, dtype) if dtype != "auto" else "auto"
        kw = dict(trust_remote_code=trust_remote_code, dtype=torch_dtype, revision=revision)
        if attn_implementation:
            kw["attn_implementation"] = attn_implementation
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4")
            kw["device_map"] = device if device != "auto" else "auto"
        elif device == "auto":
            kw["device_map"] = "auto"
        self.model = self._load(model_id, kw)
        if not load_in_4bit and device != "auto":
            self.model.to(device)
        self.model.eval()
        self._torch = torch
        if device.startswith("cuda"):
            self.meter = EnergyMeter([cuda_index(device)])
        elif device == "auto" and torch.cuda.is_available():
            self.meter = EnergyMeter(list(range(torch.cuda.device_count())))
        else:
            self.meter = EnergyMeter([])
            self.meter.ok = False

    @staticmethod
    def _load(model_id, kw):
        from transformers import AutoModelForCausalLM

        try:
            return AutoModelForCausalLM.from_pretrained(model_id, **kw)  # nosec B615: revision is in kw
        except (ValueError, KeyError):
            # natively multimodal checkpoints (e.g. Qwen3.5) register under image-text-to-text
            from transformers import AutoModelForImageTextToText

            return AutoModelForImageTextToText.from_pretrained(model_id, **kw)  # nosec B615: revision is in kw

    # ------------------------------------------------------------------ utils
    def _seed(self, seed: int) -> None:
        """Seed only this backend's device generator.

        `torch.manual_seed` reseeds every GPU in the process. When two waves run at the same
        time on different GPUs (`big_sc2`, `overlap`), a process-wide reseed from one thread
        overwrites the other GPU's generator, so both GPUs can end up drawing the same random
        stream, and identical models on identical prompts then produce identical samples.
        Seeding per device keeps concurrent waves independent."""
        t = self._torch
        if self.device.startswith("cuda") and t.cuda.is_available():
            with t.cuda.device(t.device(self.device)):
                t.cuda.manual_seed(seed)
        elif self.device.startswith("mps"):
            t.mps.manual_seed(seed)
        else:   # cpu, or device_map="auto"
            t.manual_seed(seed)

    def _sync(self):
        t = self._torch
        if self.device.startswith("cuda") or self.device == "auto":
            if t.cuda.is_available():
                t.cuda.synchronize()
        elif self.device.startswith("mps"):
            t.mps.synchronize()

    def _chat_text(self, system: str, user: str, thinking: bool | None) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        kw = {} if thinking is None else {"enable_thinking": thinking}
        try:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
        except TypeError:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def _encode(self, system: str, user: str):
        text = self._chat_text(system, user, self.enable_thinking)
        self._last_prompt_text = text
        return self.tok(text, return_tensors="pt").to(self.model.device)

    def _eos_ids(self) -> set[int]:
        ids = set()
        gc = getattr(self.model, "generation_config", None)
        e = getattr(gc, "eos_token_id", None) if gc else None
        if isinstance(e, int):
            ids.add(e)
        elif isinstance(e, (list, tuple)):
            ids.update(e)
        if self.tok.eos_token_id is not None:
            ids.add(self.tok.eos_token_id)
        return ids

    # --------------------------------------------------------------- generate
    def generate(self, system, user, n, *, temperature, top_p, max_new_tokens,
                 seed=None, question=None) -> GenResult:
        t = self._torch
        enc = self._encode(system, user)
        P = int(enc["input_ids"].shape[1])
        rec = _StatsRecorder()
        if seed is not None:
            self._seed(seed)
        self._sync()
        self.meter.start()
        t0 = time.perf_counter()
        with t.inference_mode():
            out = self.model.generate(
                **enc, do_sample=True, temperature=temperature, top_p=top_p, top_k=0,
                max_new_tokens=max_new_tokens, num_return_sequences=n,
                logits_processor=[rec], pad_token_id=self.tok.pad_token_id)
        self._sync()
        wall = time.perf_counter() - t0
        joules = self.meter.stop()
        gen = out[:, P:].cpu()
        eos = self._eos_ids()
        texts, lens, finish = [], [], []
        for row in gen.tolist():
            L = len(row)
            for i, tok in enumerate(row):
                if tok in eos:
                    L = i + 1
                    break
            hit_eos = len(row) > L or (row and row[L - 1] in eos)
            lens.append(L)
            finish.append("stop" if hit_eos else "length")
            texts.append(self.tok.decode(row[:L], skip_special_tokens=True))
        ent_steps = t.stack(rec.ent, dim=1) if rec.ent else None  # (n, steps)
        sc_steps = t.stack(rec.sc, dim=1) if rec.sc else None
        ents, scs = [], []
        for i, L in enumerate(lens):
            if ent_steps is None or L == 0:
                ents.append(None)
                scs.append(None)
                continue
            ents.append(float(ent_steps[i, :L].mean()))
            v = float(sc_steps[i, :L].mean())
            scs.append(v if math.isfinite(v) else None)
        if self.enable_thinking:
            # Budget forcing (s1-style): rows that ran out of budget are told to stop
            # thinking and answer, with a short greedy continuation.
            cut = [i for i, f in enumerate(finish) if f == "length"]
            if cut:
                t1 = time.perf_counter()
                raw = [self.tok.decode(gen[i, : lens[i]].tolist(), skip_special_tokens=False) for i in cut]
                tails = [("\n\n" if "</think>" in r else "\n</think>\n\n") + "Final answer: \\boxed{" for r in raw]
                ctx = [self._last_prompt_text + r + tl for r, tl in zip(raw, tails)]
                enc2 = self.tok(ctx, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
                with t.inference_mode():
                    out2 = self.model.generate(**enc2, do_sample=False, max_new_tokens=self.answer_tokens,
                                               pad_token_id=self.tok.pad_token_id)
                self._sync()
                wall += time.perf_counter() - t1
                cont = out2[:, enc2["input_ids"].shape[1]:].cpu().tolist()
                for j, i in enumerate(cut):
                    row = cont[j]
                    Lc = next((k + 1 for k, tok in enumerate(row) if tok in eos), len(row))
                    texts[i] = texts[i] + tails[j] + self.tok.decode(row[:Lc], skip_special_tokens=True)
                    lens[i] += len(self.tok(tails[j], add_special_tokens=False)["input_ids"]) + Lc
                    finish[i] = "forced"
        return GenResult(texts, lens, P, finish, scs, ents, [None] * n, wall, joules)

    # ----------------------------------------------------------- verification
    def score_yes(self, system: str, users: list[str], batch: int = 8,
                  questions=None) -> list[tuple[float, int]]:
        """P(Yes) / (P(Yes) + P(No)) for the next token after each prompt: one forward pass,
        no decoding. Returns (p_yes, prompt_tokens) per prompt."""
        t = self._torch
        yes_ids, no_ids = set(), set()
        for w in ("Yes", " Yes", "yes", "YES"):
            ids = self.tok.encode(w, add_special_tokens=False)
            if ids:
                yes_ids.add(ids[0])
        for w in ("No", " No", "no", "NO"):
            ids = self.tok.encode(w, add_special_tokens=False)
            if ids:
                no_ids.add(ids[0])
        yes_ids, no_ids = sorted(yes_ids - no_ids), sorted(no_ids - yes_ids)
        out = []
        for k in range(0, len(users), batch):
            texts = [self._chat_text(system, u, False) for u in users[k : k + batch]]
            enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
            with t.inference_mode():
                logits = self.model(**enc).logits[:, -1, :].float()
            p = t.softmax(logits, dim=-1)
            py, pn = p[:, yes_ids].sum(-1), p[:, no_ids].sum(-1)
            pyes = (py / (py + pn).clamp(min=1e-12)).cpu().tolist()
            plen = enc["attention_mask"].sum(-1).cpu().tolist()
            out += list(zip(pyes, [int(x) for x in plen]))
        return out

    def timed_decode(self, width, gen_len, prompt_len):
        t = self._torch
        V = len(self.tok)
        lo, hi = min(100, V // 4), min(V, 5000)
        ids = t.randint(lo, hi, (width, prompt_len), device=self.model.device)
        mask = t.ones_like(ids)
        self._sync()
        self.meter.start()
        t0 = time.perf_counter()
        with t.inference_mode():
            self.model.generate(input_ids=ids, attention_mask=mask, do_sample=False,
                                max_new_tokens=gen_len, min_new_tokens=gen_len,
                                pad_token_id=self.tok.pad_token_id)
        self._sync()
        return time.perf_counter() - t0, self.meter.stop()

    def spec(self) -> ModelSpec:
        cfg = self.model.config.to_dict()
        params = sum(p.numel() for p in self.model.parameters())
        # exclude vision tower parameters from the language-model count when present
        vis = getattr(self.model, "visual", None) or getattr(getattr(self.model, "model", None), "visual", None)
        if vis is not None:
            params -= sum(p.numel() for p in vis.parameters())
        return ModelSpec.from_hf_config(self.tag, cfg, params)

    def device_name(self) -> str:
        t = self._torch
        if self.device.startswith("cuda") and t.cuda.is_available():
            return t.cuda.get_device_name(cuda_index(self.device) or 0)
        if self.device.startswith("mps"):
            import platform

            return f"apple-mps-{platform.machine()}"
        return self.device

    def close(self):
        del self.model
        try:
            self._torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001 - best-effort cleanup
            log.debug("cuda cache not freed", extra={"error": str(e)})

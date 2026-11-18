"""Validated configuration for the live engine and server."""
from __future__ import annotations

from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ModelEntry(BaseModel):
    """One rung of the model ladder. Extra keys go to the backend constructor."""

    model_config = ConfigDict(extra="allow")
    tag: str = Field(min_length=1, pattern=r"^[A-Za-z0-9._-]+$")
    backend: Literal["hf", "mock"] = "hf"
    model_id: str | None = None
    # extra copies of this model on other devices, loaded on first use (policy `big_sc2`)
    replica_devices: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _needs_model_id(self):
        if self.backend == "hf" and not self.model_id:
            raise ValueError(f"ladder entry {self.tag!r}: backend 'hf' needs model_id")
        return self


class PolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["freewidth", "hwave", "sc", "single", "overlap", "repair", "big_sc", "big_sc2"] = "freewidth"
    tau: float = Field(0.9, gt=0.5, lt=1.0)
    max_waves: int = Field(2, ge=1, le=16)
    quorum: bool = True
    min_finished: int = Field(2, ge=1)
    width: int | Literal["free"] = "free"
    max_width: int = Field(32, ge=1, le=512)
    verify: bool = True
    verify_accept: float = Field(0.6, ge=0.0, le=1.0)
    top_c: int = Field(4, ge=1, le=16)
    sc_n: int = Field(8, ge=1, le=512)
    free_form_fallback: bool = True   # no extractable answer in wave 1 -> return best row, stop
    # --- overlap policy (two-device ladder) ---
    veto_lo: float = Field(0.2, ge=0.0, le=1.0)       # big model's P(Yes) below this vetoes an answer
    accept_hi: float = Field(0.95, ge=0.0, le=1.0)    # P(Yes) at/above this accepts an answer with >= 2 votes
    require_verified: bool = True                     # quorum accept waits until the lead has been scored
    stream_verify: bool = True                        # score each finished small-model row immediately
    speculative: bool = True                          # start the big model early on failure signals
    quorum_accept: bool = False                       # True: accept a majority B checked but is unsure about (ablation)
    trigger_min_finished: int = Field(4, ge=1)
    trigger_lead_prob: float = Field(0.6, ge=0.0, le=1.0)
    runaway_factor: float = Field(2.0, ge=1.0)        # cut rows running > factor x median finished length
    runaway_min_finished: int = Field(4, ge=1)
    verify_wait_s: float = Field(10.0, ge=0.0)        # max wait for pending verifications at wave end
    # --- repair policy (Read-Locate-Fork) ---
    flaw_p: float = Field(0.05, gt=0.0, lt=1.0)       # big model gives the small model's token < this ...
    flaw_conf: float = Field(0.5, gt=0.0, le=1.0)     # ... while preferring its own token with >= this

    @field_validator("width")
    @classmethod
    def _w(cls, v):
        if isinstance(v, int) and v < 1:
            raise ValueError("width must be >= 1 or 'free'")
        return v


class SamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    max_new_tokens: int = Field(1024, ge=1, le=32768)


class CalibrationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dir: str = "calibration"
    auto: bool = True
    eps: float = Field(0.25, gt=0.0, le=4.0)
    widths: list[int] = [1, 2, 4, 8, 16, 32]
    gen_lens: list[int] = [1, 32, 128]
    prompt_lens: list[int] = [128, 512]
    repeats: int = Field(2, ge=1, le=10)
    warmup: bool = True                  # untimed call per shape before timing (JIT kernels)
    min_r2: float = Field(0.97, ge=0.0, le=1.0)   # below this, re-measure with more repeats
    max_retries: int = Field(2, ge=0, le=5)
    typical_prompt_len: int = 160
    typical_gen_len: int | None = None   # default: 40% of max_new_tokens


class PromptConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    system: str = "You are a careful problem solver."
    template: str = ("{question}\n\nReason step by step, then give the final answer "
                     "within \\boxed{{}}.")
    apply_to_chat: bool = True

    @field_validator("template")
    @classmethod
    def _has_slot(cls, v):
        if "{question}" not in v:
            raise ValueError("prompt.template must contain {question}")
        return v


class ServerConfig(BaseModel):
    """HTTP server settings. The defaults are the safe ones: loopback only, and the server
    refuses to listen on another interface without an API key unless told otherwise."""
    model_config = ConfigDict(extra="forbid")
    host: str = "127.0.0.1"
    port: int = Field(8000, ge=1, le=65535)
    model_name: str = "freewidth"
    max_queue: int = Field(16, ge=1)
    request_timeout_s: float = Field(900.0, gt=0)
    api_key_env: str = "FREEWIDTH_API_KEY"
    allow_unauthenticated: bool = False       # serve without a key on a non-loopback host
    # request limits (each one bounds how much GPU time or memory one request can claim)
    max_body_bytes: int = Field(1_000_000, ge=1_024)
    max_messages: int = Field(64, ge=1)
    max_prompt_chars: int = Field(32_000, ge=1)
    max_samples: int = Field(64, ge=1)        # caps sc_n, width and max_width overrides
    max_waves: int = Field(4, ge=1)
    max_new_tokens: int = Field(4096, ge=1)
    allowed_policies: list[str] | None = None  # None = every policy
    public_metrics: bool = False              # /metrics without the API key
    expose_docs: bool = False                 # /docs and /openapi.json


class EngineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ladder: list[ModelEntry] = Field(min_length=1)
    policy: PolicyConfig = PolicyConfig()
    sampling: SamplingConfig = SamplingConfig()
    calibration: CalibrationConfig = CalibrationConfig()
    prompt: PromptConfig = PromptConfig()
    server: ServerConfig = ServerConfig()
    log_level: str = "INFO"
    log_json: bool = False

    @model_validator(mode="after")
    def _unique_tags(self):
        tags = [m.tag for m in self.ladder]
        if len(set(tags)) != len(tags):
            raise ValueError(f"duplicate ladder tags: {tags}")
        return self

    @classmethod
    def from_yaml(cls, path: str) -> EngineConfig:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data.get("engine", data))

"""Validation for research-pipeline configs (profile / bank / verify / analyze).

Fails fast with a readable message instead of a KeyError hours into a Kaggle run.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetCfg(_Strict):
    name: str
    split: str = "test"
    limit: int | None = Field(None, ge=1)
    seed: int = 0


class SamplingCfg(_Strict):
    temperature: float = Field(ge=0.0, le=2.0)
    top_p: float = Field(gt=0.0, le=1.0)
    max_new_tokens: int = Field(ge=1, le=65536)


class BankCfg(_Strict):
    samples_per_question: int = Field(ge=1)
    wave_width: int = Field(ge=1)
    seed: int = 1234
    keep_text: bool = False


class ProfileCfg(_Strict):
    widths: list[int] = Field(min_length=1)
    gen_lens: list[int] = Field(min_length=1)
    prompt_len: int = Field(160, ge=1)
    prompt_lens: list[int] | None = None
    typical_prompt_len: int | None = None
    repeats: int = Field(2, ge=1)


class VerifyPair(_Strict):
    small: str
    big: str


class VerifyCfg(_Strict):
    pairs: list[VerifyPair] = []
    batch: int = Field(8, ge=1)
    max_chars: int = Field(6000, ge=100)
    system: str | None = None
    template: str | None = None


class AnalysisCfg(BaseModel):
    model_config = ConfigDict(extra="allow")
    trials: int = Field(ge=1)
    bootstrap: int = Field(ge=0)
    kinetic_intensity: float = Field(gt=0)
    free_width_eps: float = Field(gt=0)
    static_batch: bool
    budget_quantiles: list[float]


class ModelCfg(BaseModel):
    model_config = ConfigDict(extra="allow")
    tag: str = Field(min_length=1, pattern=r"^[A-Za-z0-9._-]+$")
    backend: Literal["hf", "mock", "openai", "openai_compat", "vllm", "api"] = "hf"


class ResearchCfg(BaseModel):
    model_config = ConfigDict(extra="allow")   # private keys like _path
    output_dir: str
    dataset: DatasetCfg
    prompt: dict = {}
    sampling: SamplingCfg
    bank: BankCfg
    profile: ProfileCfg
    verify: VerifyCfg = VerifyCfg()
    analysis: AnalysisCfg
    models: list[ModelCfg] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self):
        tags = [m.tag for m in self.models]
        if len(set(tags)) != len(tags):
            raise ValueError(f"duplicate model tags: {tags}")
        tmpl = self.prompt.get("template")
        if tmpl is not None and "{question}" not in tmpl:
            raise ValueError("prompt.template must contain {question}")
        return self


def validate_research_config(cfg: dict) -> None:
    try:
        ResearchCfg.model_validate(cfg)
    except ValidationError as e:
        where = cfg.get("_path", "config")
        raise SystemExit(f"invalid config {where}:\n{e}") from None

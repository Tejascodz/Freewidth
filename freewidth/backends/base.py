from __future__ import annotations

from dataclasses import dataclass, field

from ..costs import ModelSpec


@dataclass
class GenResult:
    texts: list[str]
    gen_tokens: list[int]
    prompt_tokens: int
    finish: list[str] = field(default_factory=list)
    self_certainty: list[float | None] = field(default_factory=list)
    entropy: list[float | None] = field(default_factory=list)
    mean_logprob: list[float | None] = field(default_factory=list)
    wall_s: float = 0.0
    energy_j: float | None = None


class Backend:
    """A model that can draw `n` samples for one prompt in a single parallel wave."""

    tag: str = "backend"
    measures_device: bool = False   # True if wall_s reflects local hardware

    def generate(self, system: str, user: str, n: int, *, temperature: float,
                 top_p: float, max_new_tokens: int, seed: int | None = None,
                 question=None) -> GenResult:
        raise NotImplementedError

    def timed_decode(self, width: int, gen_len: int, prompt_len: int) -> tuple[float, float | None]:
        """Decode exactly `gen_len` tokens for `width` rows; return (seconds, joules)."""
        raise NotImplementedError

    def score_yes(self, system: str, users: list[str], batch: int = 8,
                  questions=None) -> list[tuple[float, int]]:
        """Prefill-only verification: P(Yes) for each prompt, plus its token count."""
        raise NotImplementedError

    def spec(self) -> ModelSpec | None:
        return None

    def device_name(self) -> str:
        return ""

    def close(self) -> None:
        pass

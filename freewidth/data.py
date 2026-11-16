"""Benchmark loading. Every dataset becomes a list of `Question`."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field


@dataclass
class Question:
    qid: str
    question: str
    answer: str
    meta: dict = field(default_factory=dict)


DEFAULT_SYSTEM = "You are a careful problem solver."
DEFAULT_TEMPLATE = (
    "{question}\n\nReason step by step, then give the final answer "
    "within \\boxed{{}}."
)


def _gsm8k(split: str, revision: str | None = None) -> list[Question]:
    from datasets import load_dataset  # lazy

    ds = load_dataset("openai/gsm8k", "main", split=split, revision=revision)  # nosec B615: pin via `revision`
    out = []
    for i, row in enumerate(ds):
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        out.append(Question(f"gsm8k-{split}-{i}", row["question"], gold))
    return out


def _math500(split: str, revision: str | None = None) -> list[Question]:
    from datasets import load_dataset  # lazy

    ds = load_dataset("HuggingFaceH4/MATH-500", split=split, revision=revision)  # nosec B615: pin via `revision`
    out = []
    for i, row in enumerate(ds):
        out.append(
            Question(
                row.get("unique_id", f"math500-{i}"),
                row["problem"],
                row["answer"],
                {"level": row.get("level"), "subject": row.get("subject")},
            )
        )
    return out


def _toy(n: int, seed: int) -> list[Question]:
    """Synthetic arithmetic with a difficulty knob; used for offline smoke runs."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        level = rng.randint(1, 5)
        terms = [rng.randint(2, 9 * level) for _ in range(level + 1)]
        q = " + ".join(map(str, terms)) + f" * {level}"
        gold = str(sum(terms[:-1]) + terms[-1] * level)
        out.append(Question(f"toy-{i}", f"Compute {q}.", gold, {"level": level}))
    return out


def _jsonl(path: str) -> list[Question]:
    out = []
    with open(path) as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            out.append(
                Question(
                    str(row.get("qid", i)),
                    row["question"],
                    str(row["answer"]),
                    {k: v for k, v in row.items() if k not in {"qid", "question", "answer"}},
                )
            )
    return out


def load_questions(spec: dict) -> list[Question]:
    name = spec["name"]
    split = spec.get("split", "test")
    limit = spec.get("limit")
    seed = spec.get("seed", 0)
    if name == "gsm8k":
        qs = _gsm8k(split, spec.get("revision"))
    elif name in ("math500", "math-500"):
        qs = _math500(split, spec.get("revision"))
    elif name == "toy":
        qs = _toy(limit or 100, seed)
    elif name.startswith("jsonl:"):
        qs = _jsonl(name.split(":", 1)[1])
    else:
        raise ValueError(f"unknown dataset {name!r}")
    if limit and len(qs) > limit:
        rng = random.Random(seed)
        qs = sorted(rng.sample(qs, limit), key=lambda q: q.qid)
    return qs


def render_prompt(q: Question, prompt_cfg: dict | None) -> tuple[str, str]:
    cfg = prompt_cfg or {}
    system = cfg.get("system", DEFAULT_SYSTEM)
    template = cfg.get("template", DEFAULT_TEMPLATE)
    return system, template.format(question=q.question)

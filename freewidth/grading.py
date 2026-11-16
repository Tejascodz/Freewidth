"""Answer extraction and equivalence checking for math-style benchmarks.

Extraction order: last \\boxed{...}, then '#### x', then 'answer is x', then the
last number in the text. Equivalence: normalised string match, then numeric
match with tolerance, then symbolic equivalence with `math-verify` (the grader used for
MATH-500 leaderboards) when it is installed, then (optionally) sympy simplification.
"""
from __future__ import annotations

import re
from fractions import Fraction

_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?")


def _last_boxed(text: str) -> str | None:
    idx = max(text.rfind("\\boxed{"), text.rfind("\\fbox{"))
    if idx < 0:
        return None
    start = text.index("{", idx) + 1
    depth = 1
    i = start
    while i < len(text) and depth:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth:
        return None
    return text[start : i - 1]


def extract_answer(text: str) -> str | None:
    if not text:
        return None
    boxed = _last_boxed(text)
    if boxed is not None and boxed.strip():
        return boxed.strip()
    m = re.findall(r"####\s*([^\n]+)", text)
    if m:
        return m[-1].strip()
    m = re.findall(r"answer is[:\s]*\$?([^\n$]+)", text, flags=re.IGNORECASE)
    if m:
        cand = m[-1].strip().rstrip(".")
        if cand:
            return cand
    nums = _NUM_RE.findall(text)
    if nums:
        return nums[-1]
    return None


_STRIP_PATTERNS = [
    (r"\\text\{([^}]*)\}", r"\1"),
    (r"\\mathrm\{([^}]*)\}", r"\1"),
    (r"\\mbox\{([^}]*)\}", r"\1"),
    (r"\\left|\\right", ""),
    (r"\\!|\\,|\\;|\\:", ""),
    (r"\\dfrac|\\tfrac", r"\\frac"),
    (r"\^\{?\\circ\}?", ""),
    (r"\\%|%", ""),
    (r"\\\$|\$", ""),
]

_UNIT_WORDS = (
    "dollars", "dollar", "cents", "cent", "units", "unit", "degrees", "degree",
    "inches", "inch", "feet", "foot", "meters", "meter", "cm", "km", "hours",
    "hour", "minutes", "minute", "seconds", "second", "days", "day", "years",
    "year", "miles", "mile", "pounds", "pound", "kg", "g",
)


def normalize(ans: str | None) -> str | None:
    if ans is None:
        return None
    s = str(ans).strip()
    for pat, rep in _STRIP_PATTERNS:
        s = re.sub(pat, rep, s)
    s = s.replace(" ", "").rstrip(".").strip()
    s = re.sub(r"^(x|y|n|k|a|b)=", "", s)
    low = s.lower()
    for w in _UNIT_WORDS:
        if low.endswith(w) and len(low) > len(w):
            s = s[: -len(w)]
            break
    # \frac{a}{b} -> a/b ; \frac ab -> a/b
    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\frac(\d)(\d)", r"\1/\2", s)
    s = s.replace("{", "").replace("}", "")
    # 1,000 -> 1000 (only when it looks like a thousands separator)
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s):
        s = s.replace(",", "")
    return s


def _to_number(s: str) -> float | None:
    t = s.replace("(", "").replace(")", "")
    try:
        if "/" in t:
            num, den = t.split("/", 1)
            return float(Fraction(num) / Fraction(den))
        return float(t)
    except (ValueError, ZeroDivisionError):
        return None


def is_equiv(pred: str | None, gold: str | None, use_sympy: bool = False) -> bool:
    if pred is None or gold is None:
        return False
    a, b = normalize(pred), normalize(gold)
    if a is None or b is None:
        return False
    if a == b:
        return True
    na, nb = _to_number(a), _to_number(b)
    if na is not None and nb is not None:
        return abs(na - nb) <= 1e-6 * max(1.0, abs(nb))
    mv = _math_verify(pred, gold)
    if mv is not None:
        return mv
    if use_sympy:
        try:  # optional, only if sympy is installed
            from sympy.parsing.latex import parse_latex  # type: ignore

            return bool((parse_latex(pred) - parse_latex(gold)).simplify() == 0)
        except Exception:
            return False
    return False


def _math_verify(pred: str, gold: str) -> bool | None:
    """Symbolic check with math-verify; None if it is not installed or cannot parse.
    Timeouts are disabled so this is safe outside the main thread."""
    try:
        from math_verify import parse, verify  # type: ignore
    except ImportError:
        return None
    try:
        g = parse(f"${gold}$", parsing_timeout=None)
        p = parse(f"${pred}$", parsing_timeout=None)
        if not g or not p:
            return None
        return bool(verify(g, p, timeout_seconds=None))
    except Exception:  # noqa: BLE001 - a grader crash is a non-match, not a failed run
        return None


def answer_key(ans: str | None) -> str | None:
    """Canonical key used to group answers when voting."""
    n = normalize(ans)
    if n is None or n == "":
        return None
    num = _to_number(n)
    if num is not None:
        return repr(round(num, 6))
    return n

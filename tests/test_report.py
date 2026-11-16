"""Statistics report: hand-checkable numbers on small synthetic runs."""
import json
import math
from pathlib import Path

import pytest

from freewidth.report import build, holm, mcnemar_n_needed, mcnemar_p, wilson


def _write(d, policy_correct, secs=None):
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "bench_rows.jsonl", "w") as f:
        for pol, correct in policy_correct.items():
            for i, c in enumerate(correct):
                s = (secs or {}).get(pol, [10.0] * len(correct))[i]
                f.write(json.dumps({"policy": pol, "qid": f"q{i}", "correct": c, "seconds": s,
                                    "joules": 100.0, "tokens": 50,
                                    "waves": [{"model": "m-4b" if pol == "big" else "m-0.8b", "tokens": 50}],
                                    "notes": {}}) + "\n")


def test_wilson_matches_reference():
    lo, hi = wilson(88, 100)
    assert lo == pytest.approx(0.8020, abs=1e-3) and hi == pytest.approx(0.9300, abs=1e-3)


def test_mcnemar_exact_and_holm():
    assert mcnemar_p(10, 2) == pytest.approx(0.03857, abs=1e-4)
    assert mcnemar_p(0, 0) == 1.0
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_power_estimate_monotone():
    small = mcnemar_n_needed(0.10, 0.02)
    smaller_effect = mcnemar_n_needed(0.07, 0.02)
    assert small is not None and smaller_effect > small
    assert mcnemar_n_needed(0.05, 0.05) is None


def test_build_report(tmp_path):
    _write(tmp_path / "r1", {"a": [1, 1, 0, 1, 0, 1, 1, 1], "big": [1, 1, 1, 1, 1, 1, 1, 0]},
           {"a": [5.0] * 8, "big": [9.0] * 8})
    path = build([str(tmp_path / "r1")], str(tmp_path / "out"), primary=[("a", "big")])
    md = Path(path).read_text()
    rep = json.loads(Path(tmp_path / "out" / "report.json").read_text())
    assert rep["largest_model"] == "m-4b"
    pol = {p["policy"]: p for p in rep["policies"]}
    assert pol["big"]["big_decode_rate"] == 1.0 and pol["a"]["big_decode_rate"] == 0.0
    (pair,) = rep["pairs"]
    assert pair["primary"] and pair["n"] == 8
    assert pair["d_mean_s"] == pytest.approx(4.0 if pair["b"] == "big" else -4.0)
    assert "**primary**" in md and (tmp_path / "out" / "accuracy_latency.png").exists()


def test_repeated_policy_is_renamed_not_merged(tmp_path):
    _write(tmp_path / "run1", {"a": [1, 0]})
    _write(tmp_path / "run2", {"a": [1, 1]})
    build([str(tmp_path / "run1"), str(tmp_path / "run2")], str(tmp_path / "out"))
    names = {p["policy"] for p in json.loads(Path(tmp_path / "out" / "report.json").read_text())["policies"]}
    assert names == {"a@run1", "a@run2"}


def test_unknown_primary_is_an_error(tmp_path):
    _write(tmp_path / "r", {"a": [1], "b": [0]})
    with pytest.raises(ValueError):
        build([str(tmp_path / "r")], str(tmp_path / "o"), primary=[("a", "zzz")])


def test_bootstrap_is_seeded(tmp_path):
    _write(tmp_path / "r", {"a": [1, 0, 1, 0, 1], "b": [1, 1, 1, 0, 0]})
    build([str(tmp_path / "r")], str(tmp_path / "o1"), seed=3)
    build([str(tmp_path / "r")], str(tmp_path / "o2"), seed=3)
    j1 = json.loads(Path(tmp_path / "o1" / "report.json").read_text())["pairs"]
    j2 = json.loads(Path(tmp_path / "o2" / "report.json").read_text())["pairs"]
    assert j1 == j2 and not math.isnan(j1[0]["d_acc_lo"])

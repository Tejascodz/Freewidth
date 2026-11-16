"""Live engine + server on the simulated ladder (no GPU)."""

import pytest

from freewidth.data import load_questions
from freewidth.engine import DeadlineExceeded, Engine, EngineConfig
from freewidth.engine.runtime import QuorumState, mock_generate_wave


def _cfg(tmp_path, **policy):
    return EngineConfig.model_validate({
        "ladder": [{"tag": "s", "backend": "mock", "params_b": 0.8},
                   {"tag": "b", "backend": "mock", "params_b": 4.0}],
        "sampling": {"max_new_tokens": 512},
        "calibration": {"dir": str(tmp_path / "calib")},
        "policy": policy,
    })


@pytest.fixture()
def engine(tmp_path):
    e = Engine(_cfg(tmp_path))
    e.load()
    yield e
    e.close()


def test_calibration_is_written_and_reused(tmp_path):
    e = Engine(_cfg(tmp_path))
    e.load()
    assert all(1 <= w <= 32 for w in e.free_widths.values())
    assert (tmp_path / "calib" / "s" / "device_fit.json").exists()
    e.close()
    e2 = Engine(_cfg(tmp_path))
    e2.load()  # second load reads the stored calibration
    assert e2.free_widths == e.free_widths


def test_config_validation_rejects_bad_values(tmp_path):
    with pytest.raises(ValueError):
        _cfg(tmp_path, tau=1.5)
    with pytest.raises(ValueError):
        EngineConfig.model_validate({"ladder": [{"tag": "x", "backend": "hf"}]})
    with pytest.raises(ValueError):
        EngineConfig.model_validate({"ladder": [{"tag": "a", "backend": "mock"}, {"tag": "a", "backend": "mock"}]})


def test_mock_wave_quorum_cuts_rows(engine):
    q = load_questions({"name": "toy", "limit": 50, "seed": 3})
    easy = min(q, key=lambda x: x.meta["level"])
    w = mock_generate_wave(engine.models["b"], "s", easy.question, 16, temperature=0.7, top_p=1.0,
                           max_new_tokens=512, seed=1, quorum=QuorumState(0.8), question=easy)
    if w.quorum_hit:
        assert any(r.finish == "quorum_cut" for r in w.rows) or all(r.finish == "stop" for r in w.rows)
        assert all(r.key is None for r in w.rows if r.finish == "quorum_cut")


def test_policies_and_trace(engine):
    qs = load_questions({"name": "toy", "limit": 30, "seed": 0})
    acc = {}
    for pol in ("single", "sc", "hwave", "freewidth"):
        ok = 0
        for i, q in enumerate(qs):
            r = engine.solve(q.question, overrides={"name": pol}, seed=i, sim_question=q)
            ok += r.answer is not None and r.answer == q.answer
            assert r.waves and r.seconds > 0 and r.tokens > 0
            if pol == "single":
                assert len(r.waves) == 1 and r.waves[0].width == 1
            d = r.to_dict()
            assert d["policy"] == pol and "text" not in d
        acc[pol] = ok / len(qs)
    assert acc["freewidth"] >= acc["single"]
    assert acc["hwave"] >= acc["single"]


def test_overrides_are_validated(engine):
    with pytest.raises(ValueError):
        engine.solve("x", overrides={"tau": 3})
    with pytest.raises(ValueError):
        engine.solve("x", overrides={"nope": 1})


def test_deadline(engine):
    import time

    with pytest.raises(DeadlineExceeded):
        engine.solve("Compute 1 + 1.", deadline=time.time() - 1)


def test_metrics_render(engine):
    engine.solve("Compute 1 + 1.", overrides={"name": "single"})
    text = engine.metrics.render()
    assert 'freewidth_requests_total{policy="single"} 1.0' in text
    assert "freewidth_request_seconds_bucket" in text


def test_every_shipped_engine_config_validates():
    import glob

    from freewidth.engine import EngineConfig

    paths = sorted(glob.glob("configs/engine_*.yaml"))
    assert len(paths) >= 5
    for p in paths:
        cfg = EngineConfig.from_yaml(p)
        assert cfg.ladder, p


def test_bench_time_budget_keeps_partial_results(tmp_path):
    import json

    from freewidth.engine import bench

    path = bench.run("configs/engine_mock.yaml", "toy", 30, ["freewidth", "big_sc"], str(tmp_path),
                     max_hours=1e-9)
    assert path is None or (tmp_path / "bench_rows.jsonl").exists()
    rows = [json.loads(x) for x in (tmp_path / "bench_rows.jsonl").read_text().splitlines()]
    assert rows == []          # budget spent before the first question: nothing run, no crash

    path = bench.run("configs/engine_mock.yaml", "toy", 12, ["freewidth", "big_sc"], str(tmp_path / "b"))
    from pathlib import Path

    md = Path(path).read_text()
    assert "| freewidth | 12 |" in md and "largest model decoded" in md
    lines = (tmp_path / "b" / "bench_rows.jsonl").read_text().splitlines()
    big = [json.loads(x) for x in lines if '"big_sc"' in x]
    assert all(r["policy"] == "big_sc" for r in big)
    assert len(big) == 12

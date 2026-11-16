"""Free-width atlas (E6): measurement, resume, sharding, failures and analysis on the simulator."""
import json
from pathlib import Path

import pytest

from freewidth import atlas


def _rows(times):
    return [{"w": w, "L": 128, "P": 128, "seconds": s} for w, s in times.items()]


def test_measured_free_width_and_censoring():
    w, cens, ratios = atlas.measured_free_width(_rows({1: 1.0, 2: 1.05, 4: 1.1, 8: 1.3, 16: 2.0}), 128, 128, 0.25)
    assert w == 4 and not cens and ratios[8] == pytest.approx(1.3)
    w, cens, _ = atlas.measured_free_width(_rows({1: 1.0, 2: 1.0, 4: 1.1}), 128, 128, 0.25)
    assert w == 4 and cens                       # widest measured wave still within budget
    assert atlas.measured_free_width(_rows({2: 1.0}), 128, 128, 0.25)[0] is None


def test_roofline_is_size_independent_and_matches_hand_calculation():
    r = {"eps": 0.25, "gpu_flops": 65e12, "gpu_bw": 320e9, "bytes_per_param": 2.0}
    assert atlas.roofline_width(r) == pytest.approx(1.25 * 65e12 / 320e9)
    assert atlas.roofline_width({**r, "bytes_per_param": 0.5}) == pytest.approx(atlas.roofline_width(r) / 4)
    assert atlas.roofline_width({**r, "gpu_flops": None}) is None


def test_collect_resumes_shards_and_analyzes(tmp_path):
    out = tmp_path / "atlas"
    atlas.collect("configs/atlas_mock.yaml", "cpu", str(out), shard=(0, 2))
    atlas.collect("configs/atlas_mock.yaml", "cpu", str(out), shard=(1, 2))
    lines = [json.loads(x) for x in (out / "atlas.jsonl").read_text().splitlines()]
    assert {r["setting"] for r in lines} == {"mock-0.5b", "mock-1b", "mock-2b", "mock-4b"}
    assert all(r["gpu_flops"] == 65e12 and r["free_width"] >= 1 for r in lines)
    n = len(lines)
    atlas.collect("configs/atlas_mock.yaml", "cpu", str(out))            # everything done: no new rows
    assert len((out / "atlas.jsonl").read_text().splitlines()) == n
    md = Path(atlas.analyze([str(out)], str(tmp_path / "an"))).read_text()
    assert "H11" in md and "H13" in md and (tmp_path / "an" / "atlas.png").exists()
    summary = json.loads((tmp_path / "an" / "atlas_summary.json").read_text())
    assert summary["by_model"]["roofline"]["median_ratio_pred_over_measured"] > 1


def test_failed_setting_is_recorded_and_retried(tmp_path, monkeypatch):
    import freewidth.backends as backends

    cfg = tmp_path / "s.yaml"
    cfg.write_text("widths: [1, 2, 4]\ngen_lens: [1, 32]\nprompt_lens: [128]\nrepeats: 1\n"
                   "operating_points: [[32, 128]]\nsettings:\n"
                   "  - {name: good, backend: mock, params_b: 1.0}\n"
                   "  - {name: bad, backend: mock, params_b: 1.0}\n")
    real = backends.make_backend

    def flaky(spec):
        if spec["tag"] == "bad":
            raise RuntimeError("CUDA out of memory")
        return real(spec)

    monkeypatch.setattr(backends, "make_backend", flaky)
    out = tmp_path / "o"
    atlas.collect(str(cfg), "cpu", str(out))
    recs = [json.loads(x) for x in (out / "atlas.jsonl").read_text().splitlines()]
    assert any(r["setting"] == "bad" and "error" in r for r in recs)
    monkeypatch.setattr(backends, "make_backend", real)
    atlas.collect(str(cfg), "cpu", str(out))                              # failed settings are retried
    recs = [json.loads(x) for x in (out / "atlas.jsonl").read_text().splitlines()]
    assert any(r["setting"] == "bad" and "error" not in r for r in recs)


def test_sweep_validation(tmp_path):
    p = tmp_path / "x.yaml"
    p.write_text("settings:\n  - {name: a, model_id: m}\n  - {name: a, model_id: m}\n")
    with pytest.raises(ValueError):
        atlas.load_sweep(str(p))
    p.write_text("settings:\n  - {name: a}\n")
    with pytest.raises(ValueError):
        atlas.load_sweep(str(p))


def test_shipped_atlas_configs_are_valid():
    for name in ("atlas_t4", "atlas_p100", "atlas_mock"):
        sw = atlas.load_sweep(f"configs/{name}.yaml")
        assert sw["settings"] and 1 in sw["widths"]

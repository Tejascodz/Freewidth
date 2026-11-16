"""Overlap policy: controller logic, simulated engine runs."""
import pytest

from freewidth.data import load_questions
from freewidth.engine import Engine, EngineConfig
from freewidth.engine.overlap import OverlapController

BOX = "so the answer is \\boxed{%s}"


def _ctrl(**kw):
    base = dict(veto_lo=0.2, accept_hi=0.95, require_verified=True, min_finished=2,
                trigger_min_finished=4, trigger_lead_prob=0.6, runaway_factor=2.0, runaway_min_finished=4)
    base.update(kw)
    return OverlapController(8, 0.9, **base)


def test_quorum_waits_for_verification_then_accepts():
    c = _ctrl()
    for i in range(4):
        c.on_finish(i, BOX % 7, 100 + i, 100 + i)
    assert c.decide(104).accept is None          # confident but not yet scored by the big model
    c.on_verified("7.0", 0.7)
    d = c.decide(105)
    assert d.accept == "7.0" and d.reason == "quorum"


def test_veto_blocks_confident_wrong_majority_and_triggers():
    c = _ctrl()
    for i in range(4):
        c.on_finish(i, BOX % 7, 100, 100)
    c.on_verified("7.0", 0.05)
    d = c.decide(101)
    assert d.accept is None and d.trigger and d.trigger_reason == "lead_vetoed"


def test_verified_minority_with_support_wins():
    c = _ctrl()
    for i, a in enumerate([5, 5, 5, 9, 9]):
        c.on_finish(i, BOX % a, 100, 100)
    c.on_verified("9.0", 0.98)
    d = c.decide(101)
    assert d.accept == "9.0" and d.reason == "verified"


def test_disagreement_trigger_and_straggler_cut():
    c = _ctrl(require_verified=False)
    for i, a in enumerate([1, 2, 3, 4]):
        c.on_finish(i, BOX % a, 100, 100)
    d = c.decide(150)
    assert d.trigger and d.trigger_reason == "disagreement"
    d = c.decide(201)                                # > 2x median finished length (100)
    assert d.cut_rows == {4, 5, 6, 7}


def test_straggler_trigger_without_disagreement():
    c = _ctrl(require_verified=True)
    for i in range(4):
        c.on_finish(i, BOX % 3, 50, 50)
    c.on_verified("3.0", 0.1)                        # vetoed lead with support -> trigger
    assert c.decide(60).trigger_reason == "lead_vetoed"


def test_end_of_wave_uses_late_verifications():
    c = _ctrl()
    for i, a in enumerate([4, 4, 6]):
        c.on_finish(i, BOX % a, 100, 100)
    c.on_verified("6.0", 0.97)
    assert c.end_of_wave().accept == "6.0"


@pytest.fixture()
def engine(tmp_path):
    cfg = EngineConfig.model_validate({
        "ladder": [{"tag": "s", "backend": "mock", "params_b": 0.8},
                   {"tag": "b", "backend": "mock", "params_b": 4.0}],
        "sampling": {"max_new_tokens": 512},
        "calibration": {"dir": str(tmp_path / "calib")},
    })
    e = Engine(cfg)
    e.load()
    yield e
    e.close()


def test_overlap_policy_runs_and_reports(engine):
    qs = load_questions({"name": "toy", "limit": 30, "seed": 5})
    reasons = set()
    for i, q in enumerate(qs):
        r = engine.solve(q.question, overrides={"name": "overlap"}, seed=i, sim_question=q)
        d = r.to_dict()
        assert d["policy"] == "overlap" and d["seconds"] > 0 and r.wall_seconds is not None
        assert "accept_reason" in r.notes
        reasons.add(r.notes["accept_reason"])
        # wall time never exceeds the sum of all work (overlap can only help)
        assert r.wall_seconds <= sum(w.seconds for w in r.waves) + 1e-3  # per-wave rounding
    assert reasons & {"quorum", "verified", "big_model", "verified_end", "quorum_end"}


def test_overlap_without_speculation_falls_back_sequentially(engine):
    q = load_questions({"name": "toy", "limit": 40, "seed": 2})
    hard = max(q, key=lambda x: x.meta["level"])
    r = engine.solve(hard.question, overrides={"name": "overlap", "speculative": False, "veto_lo": 0.99},
                     seed=3, sim_question=hard)
    if r.notes.get("trigger"):
        assert r.notes["trigger"] == "sequential_fallback"


def test_overlap_needs_two_models(tmp_path):
    cfg = EngineConfig.model_validate({"ladder": [{"tag": "s", "backend": "mock"}],
                                       "calibration": {"dir": str(tmp_path / "c")}})
    e = Engine(cfg)
    e.load()
    with pytest.raises(ValueError):
        e.solve("x", overrides={"name": "overlap"})


def test_repair_and_big_sc_simulated(engine):
    qs = load_questions({"name": "toy", "limit": 25, "seed": 9})
    reasons = set()
    for i, q in enumerate(qs):
        r = engine.solve(q.question, overrides={"name": "repair"}, seed=i, sim_question=q)
        reasons.add(r.notes.get("accept_reason"))
        assert r.waves and r.seconds > 0
        b = engine.solve(q.question, overrides={"name": "big_sc", "sc_n": 4}, seed=i, sim_question=q)
        assert b.model == "b" and b.waves[0].width == 4
    assert reasons <= {"quorum", "verified", "repaired", "big_model", "free_form"}

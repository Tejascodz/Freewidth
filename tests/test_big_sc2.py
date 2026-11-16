"""big_sc2: the big model on every device that holds a copy, one shared QuorumStop."""
import threading
from collections import Counter

import pytest

from freewidth.data import load_questions
from freewidth.engine import Engine, EngineConfig
from freewidth.engine.runtime import QuorumState, SharedQuorum


def _engine(tmp_path, replicas):
    cfg = EngineConfig.model_validate({
        "ladder": [{"tag": "s", "backend": "mock", "params_b": 0.8},
                   {"tag": "b", "backend": "mock", "params_b": 4.0, "replica_devices": replicas}],
        "sampling": {"max_new_tokens": 512},
        "calibration": {"dir": str(tmp_path / "calib")},
    })
    e = Engine(cfg)
    e.load()
    return e


def test_shared_quorum_pools_votes_across_waves():
    q = SharedQuorum(QuorumState(0.9, Counter(), 2))
    assert not q.add(["7"])                 # one vote is never a quorum
    assert not q.add([None])                # unparsed answers do not vote
    hits = []
    ts = [threading.Thread(target=lambda: hits.append(q.add(["7", "7"]))) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert q.hit and q.counts["7"] == 9


def test_big_sc2_splits_samples_over_replicas(tmp_path):
    e = _engine(tmp_path, ["sim:1"])
    qs = load_questions({"name": "toy", "limit": 20, "seed": 3})
    for i, q in enumerate(qs):
        r = e.solve(q.question, overrides={"name": "big_sc2", "sc_n": 8}, seed=i, sim_question=q)
        assert r.model == "b"
        assert [w.width for w in r.waves] == [4, 4]
        assert r.notes["split"] == [4, 4]
        assert r.seconds == pytest.approx(max(w.seconds for w in r.waves), rel=1e-3)
    info = e.info()["ladder"][1]
    assert len(info["replicas"]) == 1 and info["replica_errors"] == []
    e.release_replicas()
    assert e.info()["ladder"][1]["replicas"] == []
    r = e.solve(qs[0].question, overrides={"name": "big_sc2", "sc_n": 8}, seed=0, sim_question=qs[0])
    assert r.notes["split"] == [4, 4]       # reloaded on demand
    e.close()


def test_big_sc2_is_faster_than_one_device_at_same_samples(tmp_path):
    e = _engine(tmp_path, ["sim:1"])
    qs = load_questions({"name": "toy", "limit": 20, "seed": 4})
    one = sum(e.solve(q.question, overrides={"name": "big_sc", "sc_n": 8}, seed=i, sim_question=q).seconds
              for i, q in enumerate(qs))
    two = sum(e.solve(q.question, overrides={"name": "big_sc2", "sc_n": 8}, seed=i, sim_question=q).seconds
              for i, q in enumerate(qs))
    assert two < one
    e.close()


def test_big_sc2_without_replicas_runs_on_primary(tmp_path):
    e = _engine(tmp_path, [])
    q = load_questions({"name": "toy", "limit": 1, "seed": 1})[0]
    r = e.solve(q.question, overrides={"name": "big_sc2", "sc_n": 5}, seed=0, sim_question=q)
    assert [w.width for w in r.waves] == [5]
    e.close()


def test_replica_load_failure_is_reported_not_fatal(tmp_path, monkeypatch):
    import freewidth.engine.core as core

    e = _engine(tmp_path, ["bad"])
    real = core.make_backend

    def boom(spec):
        if spec.get("device") == "bad":
            raise RuntimeError("CUDA out of memory")
        return real(spec)

    monkeypatch.setattr(core, "make_backend", boom)
    q = load_questions({"name": "toy", "limit": 1, "seed": 2})[0]
    r = e.solve(q.question, overrides={"name": "big_sc2", "sc_n": 4}, seed=0, sim_question=q)
    assert [w.width for w in r.waves] == [4]
    assert "replica_errors" in r.notes and e.info()["ladder"][1]["replica_errors"]
    e.close()


def test_replica_devices_not_passed_to_backend(tmp_path):
    e = _engine(tmp_path, ["sim:1"])
    assert not hasattr(e.models["b"], "replica_devices")
    e.close()

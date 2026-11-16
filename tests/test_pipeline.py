import json
import os

from freewidth.cli import main


def test_smoke_pipeline(tmp_path):
    out = tmp_path / "smoke"
    assert main(["smoke", "--out", str(out)]) == 0
    a = out / "analysis"
    assert (a / "REPORT.md").exists()
    assert (a / "figures" / "frontiers.png").exists()
    assert (a / "figures" / "device_curves.png").exists()
    dis = json.loads((a / "disagreement.json").read_text())
    assert dis["kendall_tau"]
    fit = json.loads((out / "models" / "sim-0.8b" / "device_fit.json").read_text())
    assert fit["r2"] > 0.99 and fit["free_width"] >= 1
    assert (out / "models" / "sim-4b" / "verify_sim-0.8b.jsonl").exists()
    assert dis["overruns"]
    import csv

    with open(a / "results.csv") as f:
        fams = {r["family"] for r in csv.DictReader(f)}
    assert {"pverify", "short_mk", "conf_filter", "hwave"} <= fams
    from freewidth import compare

    text = compare.run([str(out), str(out)])
    assert "KV penalty ratio" in text


def test_bank_resumes(tmp_path):
    from freewidth import bank
    from freewidth.config import load_config

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_config(os.path.join(here, "configs", "smoke.yaml"))
    cfg["output_dir"] = str(tmp_path / "r")
    cfg["dataset"]["limit"] = 6
    cfg["models"] = cfg["models"][:1]
    cfg["bank"]["samples_per_question"] = 8
    assert bank.run(cfg, log=lambda *_: None)
    p = tmp_path / "r" / "models" / "sim-0.8b" / "samples.jsonl"
    n1 = len(p.read_text().splitlines())
    assert bank.run(cfg, log=lambda *_: None)   # second run adds nothing
    assert len(p.read_text().splitlines()) == n1 == 6 * 8

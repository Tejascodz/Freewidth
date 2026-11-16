"""Command line entry point.

    python -m freewidth profile  --config configs/kaggle_t4.yaml
    python -m freewidth bank     --config configs/kaggle_t4.yaml --max-hours 11
    python -m freewidth verify   --config configs/kaggle_t4.yaml
    python -m freewidth analyze  --config configs/kaggle_t4.yaml
    python -m freewidth compare  runs/t4_hybrid_math500 runs/t4_dense_math500
    python -m freewidth smoke    # full pipeline on a simulated device, no GPU needed

Live engine:
    python -m freewidth calibrate --config configs/engine_t4.yaml
    python -m freewidth solve     --config configs/engine_t4.yaml "What is 17 * 23?"
    python -m freewidth serve     --config configs/engine_t4.yaml --port 8000
    python -m freewidth bench     --config configs/engine_t4.yaml --dataset gsm8k --limit 100
    python -m freewidth report    runs/bench_final --out runs/bench_final/report
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

from .config import load_config


def _log(msg: str) -> None:
    from .logs import get_logger

    get_logger("cli").info(msg)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="freewidth")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("profile", "bank", "verify", "analyze", "all"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--only", nargs="*", help="model tags to include")
        p.add_argument("--max-hours", type=float, default=None)
    s = sub.add_parser("smoke")
    s.add_argument("--out", default="runs/smoke")
    sv = sub.add_parser("serve", help="run the OpenAI-compatible engine server")
    sv.add_argument("--config", required=True)
    sv.add_argument("--host", default=None)
    sv.add_argument("--port", type=int, default=None)
    sv.add_argument("--insecure-no-auth", action="store_true",
                    help="allow serving without an API key on a non-loopback address")
    cb = sub.add_parser("calibrate", help="measure free width for every ladder model")
    cb.add_argument("--config", required=True)
    cb.add_argument("--force", action="store_true", help="re-measure even if a calibration exists")
    be = sub.add_parser("bench", help="live benchmark of engine policies on a dataset")
    be.add_argument("--config", required=True)
    be.add_argument("--dataset", default="gsm8k")
    be.add_argument("--split", default="test")
    be.add_argument("--limit", type=int, default=100)
    be.add_argument("--policies", default="single,sc,hwave,freewidth")
    be.add_argument("--out", default="runs/bench")
    be.add_argument("--max-hours", type=float, default=None,
                    help="stop starting new questions after this many hours; results so far are kept")
    be.add_argument("--seed", type=int, default=0)
    so = sub.add_parser("solve", help="answer one question with the live engine")
    so.add_argument("--config", required=True)
    so.add_argument("question")
    so.add_argument("--policy", default=None)
    so.add_argument("--json", action="store_true")
    for p_ in (s, sv, cb, so, be):
        p_.add_argument("--log-json", action="store_true")
    for p_ in sub.choices.values():
        if not any(a.dest == "log_level" for a in p_._actions):
            p_.add_argument("--log-level", default="INFO")
    rp = sub.add_parser("report", help="statistics report for one or more bench runs")
    rp.add_argument("runs", nargs="+", help="bench output directories or bench_rows.jsonl files")
    rp.add_argument("--out", required=True)
    rp.add_argument("--title", default=None)
    rp.add_argument("--ladder", default=None, help="comma-separated model tags, smallest first")
    rp.add_argument("--seed", type=int, default=0)
    rp.add_argument("--primary", default=None,
                    help="pre-declared comparisons, e.g. overlap:big_sc2,overlap:freewidth")
    c = sub.add_parser("compare")
    c.add_argument("runs", nargs="+")
    c.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    from .logs import setup_logging

    setup_logging(getattr(args, "log_level", "INFO"), getattr(args, "log_json", False))

    if args.cmd == "report":
        from .report import build

        path = build(args.runs, args.out, args.title,
                     args.ladder.split(",") if args.ladder else None, args.seed,
                     [tuple(x.split(":")) for x in args.primary.split(",")] if args.primary else None)
        with open(path) as f:
            print(f.read())
        return 0

    if args.cmd == "serve":
        from .engine.server import serve

        serve(args.config, args.host, args.port, args.log_json, args.log_level, args.insecure_no_auth)
        return 0
    if args.cmd == "bench":
        from .engine import bench

        print(bench.run(args.config, args.dataset, args.limit,
                        [p.strip() for p in args.policies.split(",") if p.strip()],
                        args.out, args.seed, args.split, args.max_hours))
        return 0
    if args.cmd in ("calibrate", "solve"):
        from .engine import Engine

        eng = Engine.from_yaml(args.config, load=False)
        if args.cmd == "calibrate" and args.force:
            shutil.rmtree(eng.cfg.calibration.dir, ignore_errors=True)
        eng.load()
        try:
            if args.cmd == "calibrate":
                for row in eng.info()["ladder"]:
                    print(f"{row['tag']:>20}  device={row['device']}  free_width={row['free_width']}  "
                          f"R2={row['calibration_r2']}")
                return 0
            ov = {"name": args.policy} if args.policy else None
            res = eng.solve(args.question, overrides=ov)
            if args.json:
                print(json.dumps({**res.to_dict(), "text": res.text}, indent=2))
            else:
                print(res.text)
                print(f"\n--- answer={res.answer} confidence={res.confidence} model={res.model} "
                      f"waves={len(res.waves)} tokens={res.tokens} seconds={res.seconds:.2f} "
                      f"escalated={res.escalated} verified={res.verified}")
            return 0
        finally:
            eng.close()

    if args.cmd == "compare":
        from . import compare

        print(compare.run(args.runs, args.out))
        return 0

    if args.cmd == "smoke":
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cfg = load_config(os.path.join(here, "configs", "smoke.yaml"))
        cfg["output_dir"] = args.out
        shutil.rmtree(args.out, ignore_errors=True)
        from . import analyze, bank, profiler, verify

        profiler.run(cfg, log=_log)
        bank.run(cfg, log=_log)
        verify.run(cfg, log=_log)
        analyze.run(cfg, log=_log)
        return 0

    cfg = load_config(args.config)
    os.makedirs(cfg["output_dir"], exist_ok=True)
    shutil.copy(cfg["_path"], os.path.join(cfg["output_dir"], "config.yaml"))
    if args.cmd in ("profile", "all"):
        from . import profiler

        profiler.run(cfg, args.only, log=_log)
    if args.cmd in ("bank", "all"):
        from . import bank

        done = bank.run(cfg, args.only, args.max_hours, log=_log)
        if not done:
            _log("bank incomplete (time budget); re-run the same command to resume")
            if args.cmd == "all":
                return 2
    if args.cmd in ("verify", "all"):
        from . import verify

        verify.run(cfg, log=_log)
    if args.cmd in ("analyze", "all"):
        from . import analyze

        analyze.run(cfg, log=_log)
    return 0


if __name__ == "__main__":
    sys.exit(main())

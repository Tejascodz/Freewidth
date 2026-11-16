# Architecture

```
freewidth/
├── grading.py        answer extraction + equivalence (boxed, ####, numbers, fractions)
├── data.py           GSM8K, MATH-500, toy, jsonl -> Question
├── costs.py          ModelSpec, FLOPs / eFLOPs pricing, DeviceFit T(w, L, P), free width
├── hw.py             NVML energy meter
├── backends/         one interface, four implementations
│   ├── base.py         generate (one wave), timed_decode, score_yes, spec
│   ├── hf.py           transformers: waves, self-certainty, budget forcing, prefill P(Yes)
│   ├── openai_compat.py  vLLM / llama.cpp / Ollama / free-tier APIs
│   └── mock.py         simulated device for tests and smoke runs
├── profiler.py       measure + fit the device cost curve
├── bank.py           resumable sample bank
├── verify.py         prefill verification pass over a bank
├── replay.py         load banks, sample streams, price traces three ways
├── strategies.py     12 strategy families incl. hwave, QuorumStop, pverify
├── pareto.py         frontiers, Kendall tau, Jaccard, fits, plots
├── analyze.py        replay grid -> results.csv, disagreement.json, REPORT.md
├── compare.py        cross-run table (H2)
├── report.py         live-run statistics: Wilson CIs, McNemar + Holm, paired bootstrap, power
├── schema.py         validation for research configs
├── logs.py           text / JSON logging
├── engine/
│   ├── config.py       validated engine config
│   ├── runtime.py      live waves with a QuorumStop stopping criterion
│   ├── core.py         Engine: ladder, calibration, policies, traces
│   ├── metrics.py      Prometheus counters / gauges / histograms
│   ├── server.py       FastAPI app, OpenAI-compatible
│   └── bench.py        live benchmark of policies
└── cli.py            profile, bank, verify, analyze, all, compare, smoke,
                      calibrate, solve, serve, bench
```

**Research path:** profile → bank → verify → analyze → compare. Everything expensive runs once; every strategy is then a cheap replay.

**Production path:** calibrate → serve. The same ideas run online: free width comes from the same cost-curve fit, QuorumStop runs as a transformers `StoppingCriteria`, and verification is the same prefill `score_yes`.

**Shared contract.** A *wave* is `n` parallel samples of one prompt on one device. Strategies, the replay pricer and the live engine all reason in waves, which is what makes the offline predictions comparable with the live benchmark.

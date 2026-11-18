# Changelog

All notable changes to this project. Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## 0.8.0
- `overlap` now escalates a small-model majority the verifier is unsure about, instead of accepting it (`quorum_accept: false` by default). The old behaviour is available as `quorum_accept: true`.
- The README describes the current system and results only; `scripts/make_figures.py` regenerates the results figure.

## 0.7.x
- Results on 300 GSM8K questions for `overlap`, `big_sc2` and `repair`, including the acceptance-rule ablation.
- Free-width atlas: `freewidth atlas collect|analyze`, sweep configs for T4 and P100, held-out validation of the cost model.
- Literature review and positioning against prior work.
- Kaggle scripts run end to end in CI on a simulated device.
- Fixes: report generation with no primary comparisons, atlas resume across GPUs, one output file per GPU worker.

## 0.6.x
- Policy option `quorum_accept`.
- The README was restructured as a technical report.

## 0.5.x
- Baseline `big_sc2`: the large model on every GPU, with one shared QuorumStop.
- `freewidth report`: Wilson intervals, exact McNemar tests with Holm correction, paired bootstrap, power analysis.
- `bench --max-hours` time budget; MATH-500 grading with math-verify.
- Security hardening:
  - loopback by default; unauthenticated public binds refused; constant-time API key check;
  - request limits; no `trust_remote_code`; non-root Docker;
  - pinned CI, with bandit, pip-audit, detect-secrets and CodeQL.
- Per-device RNG seeding for concurrent waves.

## 0.4.x
- Policies `repair` (Read-Locate-Fork) and `big_sc`.
- The verifier reads candidates as third-party solutions.

## 0.3.0
- Policy `overlap`: two-GPU ladder with streaming verification, veto, verified accept, speculative escalation and straggler cuts.

## 0.2.x
- Live engine with free-width calibration, in-wave QuorumStop and prefill verification.
- OpenAI-compatible server with auth, queueing, deadlines and metrics.
- Commands `calibrate`, `solve`, `serve`, `bench`.
- Calibration warm-up and automatic re-measurement when the fit is poor.

## 0.1.0
- Research pipeline:
  - device profiling and a resumable sample bank;
  - a prefill verification pass;
  - offline replay of 12 strategy families under FLOPs, eFLOPs and measured device cost.

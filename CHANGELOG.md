# Changelog

## 0.7.3 (docs: systematic novelty review)
- **Literature review.** Covered about 90 papers across eight components. The README now has a verdict for each component, a narrow statement of contribution, the related work a reviewer will expect, and about 30 new references.
- **Newly cited closest prior work:**
  - DUCHESS, short-m@k, First Finish Search and SART for QuorumStop;
  - SpecReason, GenRM, ATTS and mixture-of-thought cascades for `overlap`;
  - Khullar et al. (self-attribution bias) for the verifier framing result.
- **Corrections:**
  - QuorumStop's "at least 2 finished rows" guard never binds at τ = 0.9; the earliest stop is three agreeing rows.
  - p_lead is described as a heuristic score, not a calibrated probability.
  - The verifier framing result confounds placement with wording, so it is no longer called self-attribution bias.
  - The cost model's attention term is a dense-model assumption.
- **Missing baselines** that a reviewer will expect are listed: short-m@k, a DUCHESS-style quorum, an ATEST-style stopper, a sequential cascade, a single-GPU version of `overlap`, an ε sweep, CUDA graphs, and a placement-only verifier ablation.

## 0.7.2 (pre-flight fixes before H10 and the atlas)
- **Fix:** `kaggle_experiment.sh` treated `PRIMARY=""` as unset and substituted the default comparisons. The H10 run has only `overlap`, so its report would have failed at the end of the run. `PRIMARY=""` now means no primary comparisons.
- **Fix:** a benchmark crash no longer aborts the script before the calibration copy and the report; rows finished so far are kept either way.
- **Fix (atlas resume):** a setting already measured in the output folder is skipped whichever GPU measured it. The pilot and the full sweep shard settings differently, so the full run would have re-measured the pilot's settings on the other GPU.
- **Fix (atlas):** each GPU worker writes its own `atlas.<device>.jsonl`, so the two concurrent workers never write to the same file. A line cut off by a killed session is skipped. A failed setting that later succeeded is not reported as failed. The scripts create their output folder.
- **Tests:** the Kaggle scripts themselves now run end to end in CI on the simulated device, with GPU tools stubbed (`tests/test_scripts.py`). The `PRIMARY` test fails on the old script.
- **Pre-registration:** E1b's exact analysis command (run names, primary pairs) is fixed in Appendix A before any H10 data. The known small-wave inefficiency after an escalation is documented and deliberately left unchanged.

## 0.7.1
- **Wording.** "Near-free" now always means "within the 1.25× latency budget"; the README no longer says extra samples "cost almost nothing".
- **R² qualified as in-sample.** New out-of-sample checks in the atlas (pre-registered as H15):
  - the cost curve is fitted on widths ≤ 16 and scored on widths 32–64 (`holdout_mape`);
  - the free width from the fitted curve is compared with the one read directly from timings.
- **Pilot sweep** `configs/atlas_t4_pilot.yaml` (4 settings, about 1 h, a subset of the full sweep with the same names, so the full run resumes from it). `kaggle_atlas_t4.sh` takes `CONFIG`.

## 0.7.0
- **Literature review.** A systematic search found prior work that the earlier novelty statement missed:
  - near-free parallelism (He et al. 2026), which uses the same latency criterion for decode positions;
  - batch-saturation studies and latency predictors: *Mind the Memory Gap*, SLIM, Vidur, KernelSight-LM;
  - edge parallel test-time scaling: EdgeReasoning, Efficient Reasoning on the Edge, mobile-NPU TTS, FastTTS;
  - resource-aware TTS: ATEST, ATTS, *Sample Count Is Not Enough*;
  - in-batch pruning: ST-BoN, Speculative Rejection.

  The README's novelty statement, related work and references are rewritten. The remaining contribution is stated as a pre-registered systems evaluation plus measurements on hybrid models on consumer GPUs.
- **Free-width atlas (E6).**
  - `python -m freewidth atlas collect|analyze`: calibration-only sweeps over models, number formats, sequence lengths and GPUs. Free width is measured directly from timings, with censoring. Results are written per measurement and resumable, with a time budget.
  - Analysis: roofline comparison, size trend, leave-one-model-out predictors.
  - Configs `atlas_t4.yaml`, `atlas_p100.yaml`, `atlas_mock.yaml`; scripts `kaggle_atlas_t4.sh` (one process per GPU) and `kaggle_atlas_p100.sh`.
  - H11–H14 are pre-registered in Appendix A. There are tests on the simulated device.

## 0.6.1
- README rewritten as a technical report: abstract, problem, the free-width cost model and measurement, system design (QuorumStop, prefill verification, policies and decision rules), evaluation methodology and statistical protocol, results with pre-registered tests and error analysis, discussion, limitations, reproduction, engineering, related work and references. The supporting material moved to appendices A–E. No emojis.

## 0.6.0
- **H10 switch:** `quorum_accept: false` (policy option, also a per-request override). When the big model has checked the small model's majority but is unsure (veto_lo ≤ P(Yes) < accept_hi), overlap escalates at that moment (trigger `uncertain_quorum`) instead of accepting. Comes with controller and policy tests.
- E1b pre-registered precisely. `configs/engine_t4_h10.yaml` and `scripts/kaggle_h10.sh` (session capped at 5 h to protect the weekly GPU quota).
- `kaggle_experiment.sh`: `PRIMARY=""` runs the report without primary comparisons.
- All engine configs bind to `127.0.0.1`; Docker passes `--host 0.0.0.0` with a required API key.
- **README redesigned:** the idea in one picture, results with honest reading, a mermaid diagram of the overlap ladder, quickstart, and a "what worked, what failed" table. The research background moved to `docs/research.md`.
- **One README:** all documentation pages (results, experiments, engine, research design, limitations, architecture) are merged into README.md as collapsible sections. `docs/` keeps only figures. SECURITY.md, CONTRIBUTING.md and CHANGELOG.md stay separate because GitHub uses them.
- Wording fix: E1's 28 `quorum` accepts were majorities the 4B had checked and was *unsure* about, not unchecked ones.

## 0.5.2
- E1 complete: rows, report and write-up for all three policies (overlap, big_sc2, repair) on 300 questions. Both pre-registered claims failed. The overlap gap is traced to 28 unverified quorum accepts, and H10 / E1b are pre-registered.
- Correction: the data shows the 0.5.1 seeding bug did **not** duplicate `big_sc2`'s samples (wave token counts match on 11 of 300 questions). The fix stays.
- `report` / `bench`: model tags are normalised (`q35-4b (cancelled)`, `q35-4b.r1` → `q35-4b`), so "largest model decoded" is correct. It previously read 0% for `big_sc2` and `repair`.

## 0.5.1 (security hardening)
- **Server**
  - Binds to `127.0.0.1` by default, and refuses to listen on any other address without an API key unless `--insecure-no-auth` / `server.allow_unauthenticated` is set.
  - API key compared in constant time (`hmac.compare_digest`). `WWW-Authenticate` on 401.
  - New request limits: body size (Content-Length required), message count, prompt characters, samples (`sc_n`/`width`/`max_width`), waves and new tokens. Optional policy allow-list. `seed` is bounded.
  - `/health` returns liveness only. Engine details moved to the authenticated `/v1/info`. `/metrics` needs the key unless `server.public_metrics` is set.
  - OpenAPI docs are off by default. Responses carry `nosniff` and `no-store`. No `Server` header. 500s carry only the request id.
  - Client request ids are validated against `[A-Za-z0-9._-]{1,64}`.
- **Bug fix, concurrent sampling:**
  - Every wave used to call `torch.manual_seed`, which reseeds all GPUs. Two waves started at the same moment on two GPUs (`big_sc2`) could therefore end up drawing the same random stream, and identical models would produce identical samples. E1's `big_sc2` was most likely affected.
  - Each backend now seeds only its own device (`HFBackend._seed`). There is a test.
- **Model loading:** `trust_remote_code` now defaults to **off** (it was on). Optional `revision:` pin per model; unpinned loads log a warning. Datasets accept a `revision` too.
- **OpenAI-compatible backend:** only `http(s)` URLs. It refuses to send an API key over plain `http` to a non-loopback host unless `allow_insecure_http: true` is set.
- **Logging:** text logs escape line breaks, so a value cannot forge log lines. Silent `except: pass` cleanup blocks now log at debug level.
- **Docker:** runs as an unprivileged user. Binds inside the container need the API key. Health check on 127.0.0.1. Added a `.dockerignore`.
- **CI**
  - Actions pinned to commit SHAs, `permissions: contents: read`, no persisted checkout credentials, job timeouts.
  - New `security` workflow: bandit (fails on medium severity and above), pip-audit, detect-secrets and CodeQL (security-extended), on every push and weekly.
  - Dependabot for pip, Actions and Docker.
- **Docs and templates:** `SECURITY.md` (reporting and threat model), `CONTRIBUTING.md`, PR and issue templates. `.gitignore` excludes `.env*`, keys and `kaggle.json`.

## 0.5.0
- New baseline policy `big_sc2`: the largest model on every device that holds a copy
  (`replica_devices` in its ladder entry, loaded on first use). Samples are split evenly and all
  waves share one QuorumStop (`SharedQuorum`), so every device stops when the pooled votes are
  confident. In earlier runs, `big_sc` used one GPU and left the other idle, which favoured the
  two-GPU policies on latency.
- Replica loading and mid-request out-of-memory errors degrade to fewer devices instead of failing,
  and are reported in `/v1/info` and in the request notes.
- `python -m freewidth report`: statistics report for one or more runs:
  - Wilson intervals; exact McNemar tests with Holm correction within a pre-declared primary family;
  - paired-bootstrap intervals for accuracy and latency differences;
  - McNemar sample size for 80% power;
  - breakdowns by engine decision, and an accuracy-latency figure.
- MATH-500 answers are graded with math-verify when it is installed, falling back to string and number matching.
- New configs: `engine_t4_math.yaml` (MATH-500, 2048 tokens), `engine_t4_2b.yaml` (Qwen3.5-2B -> 4B),
  `engine_p100.yaml` (one P100); `replica_devices` added to the dense config.
- `scripts/kaggle_experiment.sh`: one pre-registered experiment per commit, with the report built at the end.
- `docs/experiments.md` (pre-registered experiments E1–E5), `docs/limitations.md`, `CITATION.cff`.
- Run 2's overlap vs freewidth result is restated with multiple-comparison correction
  (p = 0.039 uncorrected, 0.077 after Holm over the two primary comparisons).
- `bench --max-hours`: stops starting new questions when the budget is spent, and still writes every
  finished row and the report (a partial policy is marked as such). The Kaggle scripts set it from the
  session clock (11.2 h), flush every row to disk, and show progress every 10 questions.
- `bench`: "largest model decoded" is now measured from the waves themselves. It was previously
  escalation or trigger, which reported 0 for `big_sc`/`big_sc2`.
- E1 results (partial): `docs/results-t4-gsm8k-E1.md`. The pre-registered overlap vs big_sc2 claim failed.
- `bench` unloads replicas after each policy, so policies that follow `big_sc2` get the memory back.
- `scripts/kaggle_final.sh`: overlap, big_sc2 and repair (0.4.1 fix) in one session on 300 GSM8K
  questions (a superset of the 100 used before).

## 0.4.1
- `repair`: the big model now reads the candidate as a third-party solution in a grading prompt
  (verdict + flaw from the same single pass). In 0.4.0 the candidate was framed as the big model's
  own assistant turn, which pushed P(Yes) to ~0.99 for almost every candidate on the T4 run
  (self-correction blind spot) and made `repair` accept wrong answers.
- Results of the 4-way T4 run in `docs/results-t4-gsm8k-4way.md` and `results/`.

## 0.4.0
- New engine policy `repair` (Read-Locate-Fork): one big-model forward pass returns both the
  verdict P(Yes) and the first flawed solution token; the big model's token replaces it and the
  small model repairs in a free-width batch with QuorumStop; big-model decoding only as last resort.
- New baseline policy `big_sc`: majority vote over samples of the largest ladder model.
- `bench` writes BENCH.md / CSV after every policy, so partial runs keep finished results.
- `scripts/kaggle_4way.sh`: overlap, repair, freewidth, big_sc on the same 100 GSM8K questions.
- README positioning corrected: latency-aware test-time scaling (EMNLP Findings 2025) already
  showed compute-optimal is not latency-optimal; FreeWidth's measured per-device free width on
  single-user hardware is the narrower claim.

## 0.3.0
- New engine policy `overlap`: two-device ladder with streaming cross-model verification,
  veto of confident-but-wrong majorities, verified accept, speculative escalation triggered
  by wave dynamics (disagreement, vetoed lead, stragglers), and per-row straggler cuts.
  Live path uses a transformers StoppingCriteria plus verifier and big-model threads; mock
  path is an event simulation on the fitted device curves.
- `SolveResult.wall_seconds` and `notes` (trigger, accept reason, verified/cut rows);
  Prometheus counters for big-model starts and accept reasons.
- Results of the first live T4 run in `docs/results-t4-gsm8k.md` and `results/`.

## 0.2.2
- Calibration re-measures automatically (more repeats) when the fit R² is below
  `calibration.min_r2` (default 0.97); a poor stored calibration is never reused.
- `bench` records each model's free width and calibration R² in BENCH.md and engine_info.json.

## 0.2.1
- Calibration: untimed warm-up per (width, prompt length) so JIT kernels (Triton /
  flash-linear-attention) compile outside the timed region; default 2 repeats.
- Use `dtype=` when loading models (removes the transformers deprecation warning).

## 0.2.0
- Live engine: model ladder with free-width calibration, online QuorumStop inside each wave,
  prefill-verified escalation, policies `freewidth`, `hwave`, `sc`, `single`.
- OpenAI-compatible server (`/v1/chat/completions`, `/v1/models`, `/health`, `/metrics`) with
  API-key auth, bounded queue, request deadlines, request ids and SSE compatibility.
- `calibrate`, `solve`, `serve`, `bench` commands.
- Validated configs (pydantic) for both the research pipeline and the engine.
- Structured logging (`--log-json`), ruff lint, GitHub Actions CI, Dockerfile, Makefile.

## 0.1.0
- Research harness: device profiling, resumable sample bank, prefill verification pass,
  offline replay of 12 strategy families under FLOPs / eFLOPs / measured device cost,
  iso-accuracy overrun, cross-run comparison.

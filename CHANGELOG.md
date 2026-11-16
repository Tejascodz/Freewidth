# Changelog

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

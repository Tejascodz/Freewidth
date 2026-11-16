# FreeWidth: Hardware-Conditional Test-Time Compute Scaling

[![ci](https://github.com/Tejascodz/freewidth/actions/workflows/ci.yml/badge.svg)](https://github.com/Tejascodz/freewidth/actions/workflows/ci.yml)
[![security](https://github.com/Tejascodz/freewidth/actions/workflows/security.yml/badge.svg)](https://github.com/Tejascodz/freewidth/actions/workflows/security.yml)
![python](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue)
![license](https://img.shields.io/badge/license-MIT-green)

**Question.** A small model can spend extra inference compute in three ways:

- **Think wider:** sample many answers and vote.
- **Think longer:** reason for more tokens, or sample one answer at a time until confident.
- **Hand off:** pass the question to a bigger model, which can either decode its own answer or only read and check the small model's answers.

Which choice is compute-optimal? Published answers rank these options with **one** cost model, usually FLOPs or datacenter memory traffic.

**Core claim under test.** The ranking depends on where the model runs:

- **Multi-tenant datacenter.** Weight reads are amortised across many users, so every sample pays its full compute and KV traffic.
- **Single-tenant device** (a T4, a laptop, an edge box). Decoding is bound by memory bandwidth, so a wave of *w* parallel samples reads the weights once per step. Up to a hardware-specific **free width**, the extra samples are almost free.

FreeWidth has two parts:

- **Research pipeline.** It measures the device, generates one reusable bank of samples, replays every strategy offline, and prices each replay under three cost models (FLOPs, datacenter eFLOPs, measured on-device seconds and joules).
- **Live engine.** An OpenAI-compatible server and Python library that run the proposed methods for real on a local model ladder: free-width waves, live QuorumStop, and prefill-verified escalation. See [docs/engine.md](docs/engine.md).

```bash
python -m freewidth serve --config configs/engine_t4.yaml        # GPU, listens on 127.0.0.1:8000
python -m freewidth serve --config configs/engine_mock.yaml      # try it without a GPU
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"freewidth","messages":[{"role":"user","content":"What is 17 * 23?"}]}'
```

### Live results (Kaggle T4 x2, GSM8K)

**E1: 300 questions, same session** ([write-up](docs/results-t4-gsm8k-E1.md)):

| policy | accuracy (95% CI) | median s | mean s | mean J |
|---|---|---|---|---|
| **overlap** (0.8B → 4B ladder across two GPUs) | 89.3% (85.3–92.3) | **19.3** | **34.5** | **2,849** |
| **big_sc2** (4B alone on both GPUs) | **95.7%** (92.7–97.5) | 40.2 | 43.4 | 5,713 |

- **The pre-registered claim failed.** Overlap was not "faster at equal accuracy".
- **What the run shows instead is a trade-off.** Overlap gives half the typical latency and half the energy, for about 6 points less accuracy. Exact paired tests are pending the per-question rows.
- **The two-GPU baseline is probably slower than it could be** (Python threads, not processes), so the trade-off may favour the 4B even more. The process-per-GPU fix is next.
- **Repair was cut off** by Kaggle's 12-hour limit and is still untested.

**Runs 1–2: 100 questions.** These earlier runs used a one-GPU 4B baseline that left the other GPU idle, which made the comparison unfair ([run 1](docs/results-t4-gsm8k.md), [run 2](docs/results-t4-gsm8k-4way.md)):
- freewidth 80%, overlap 88%, 4B alone 93%;
- repair 71% (verifier framing bug, since fixed).

---|---|---|---|---|
| single | 0.53 | 22.1 | — | 915 |
| sc (8 samples) | 0.68 | 38.5 | 45.5 | 2,529 |
| hwave | 0.79 | 48.5 | 19.1 | 3,232 |
| **freewidth** | **0.78** | **41.5** | **19.2** | **2,713** |

`freewidth` beats `sc` by 10 points (paired McNemar p = 0.04) at a statistically indistinguishable mean time. It matches `hwave` accuracy while being 7 s faster per question and cutting big-model escalations from 18% to 7%. Full write-up, caveats and raw data: [docs/results-t4-gsm8k.md](docs/results-t4-gsm8k.md).

---

## Coverage map: gaps found, and how this repo addresses them

| # | Gap in the literature (as of November 2026) | Closest prior work | What FreeWidth does |
|---|---|---|---|
| G1 | Frontiers are computed under one cost model. Latency-aware work shows compute-optimal is not latency-optimal on datacenter GPUs, but no measured per-device "free width" on single-user consumer hardware | Snell et al. 2024; Kinetics 2025; UniScale 2026; Latency-Aware Test-Time Scaling (EMNLP Findings 2025) | Prices identical traces under FLOPs, multi-tenant eFLOPs and a cost curve measured on the device. Reports Kendall tau, Pareto overlap, and a new **iso-accuracy overrun** metric (how much slower you run if you trust the wrong frontier) |
| G2 | Kinetics' conclusion that small models are over-estimated assumes full attention in every layer | Kinetics (dense Qwen3 models) | Runs hybrid Qwen3.5 (Gated DeltaNet) against a dense Qwen3 control on the same device; `compare` reports the KV-penalty ratio (H2) |
| G3 | Parallel early-stop methods use a fixed *m* (short-m@k) or per-trace confidence (DeepConf), priced in tokens | Hassid et al. 2025; Fu et al. ICLR 2026; Aggarwal et al. 2023 | **QuorumStop:** a posterior-adaptive stop over the order in which rows finish inside a lock-step wave, priced on static-batch hardware, where the slowest row sets the cost (H3) |
| G4 | Verification is priced as decoding (GenRM verification chains), so self-consistency wins under FLOPs | Singhi et al. COLM 2025; GenRM; SplitReason 2025 (uses prefill to decide hand-offs, not to verify answers) | **Prefill-Verified Escalation:** the big model never decodes; it scores the small model's distinct candidates in one forward pass. Tests whether "self-consistency beats verification" reverses under device cost (H4) |
| G5 | "Think longer" (thinking-mode budgets) and "think wider" are rarely compared on one measured-cost frontier | s1 budget forcing (Muennighoff et al. 2025); short-m@k | Thinking-budget variants (budget forcing) are banked as extra model tags and replayed on the same frontier (H5) |
| G6 | Answer-level cascades run sequentially, so on a multi-GPU single-user box the big model's device sits idle while the small model decodes, and escalated questions pay the small model's full time first | Speculative cascades (token level, Narasimhan et al. 2025); ConfSpec (step level, ACL 2026); MSV streaming verifier (single verifier, selection only, 2026); MARS mid-generation stopping (one model, simulated, 2026); Signed Rescue Routing (sequential, learned router, 2026) | **Overlap** (engine policy `overlap`): streaming cross-model verification with veto, verified accept, and speculative escalation triggered by the wave's own dynamics (disagreement, vetoed lead, stragglers), on separate GPUs. Measured live, not simulated |
| G7 | When a small model fails, either the big model writes a whole new answer (cascades) or someone locates the error with the same model or a separate scorer (StepBack) | StepBack (2026); Shen et al. *Reasoning Can Be Restored by Correcting a Few Decision Tokens* (2026); SpecAgent (2026) | **Read-Locate-Fork** (engine policy `repair`): one big-model forward pass gives both the verdict and the first flawed token; the big model writes one token; the small model repairs in a parallel free-width batch |

The literature check behind this table found **short-m@k** and **DeepConf** already published, so both are included as **baselines**, not claimed as new. Check arXiv again before calling anything a first. Searches cannot prove that something was never tried; they can only show that it was not found.

**Not covered (deliberately):** PRM-guided beam or tree search, multi-agent debate, and RL-trained adaptive-length models. All three need online generation with branching, which offline bank replay cannot reproduce. They are listed as future work.

## Hypotheses (each one can be falsified)

- **H1, regime flip.** Kendall tau(FLOPs, device seconds) is well below 1, the Pareto frontiers overlap little (Jaccard < 0.5), and the worst iso-accuracy overrun (FLOPs to device) is above 2×. *Refuted if* the frontiers match.
- **H2, hybrids weaken Kinetics.** The KV-penalty ratio (small model sampling wide vs. big model answering once, under eFLOPs) is lower for hybrid Qwen3.5 than for dense Qwen3 at similar sizes.
- **H3, QuorumStop.** At matched accuracy, `hwave(quorum=True)` costs fewer device seconds than both `short_mk` and full-wave voting, because wrong chains tend to be the long stragglers.
- **H4, prefill verification.** Under device seconds, `pverify` beats both self-consistency on the small model and decoding with the big model at matched accuracy. This reverses Singhi et al.'s FLOPs-based conclusion for a single-tenant device.
- **Status after E1:**
  - H6 and H7 supported (run 2).
  - H8 not yet tested: the 0.4.0 read-framing bug, then the E1 time limit.
  - H9 refuted: the 4B alone on both GPUs is 6.4 points more accurate than overlap on 300 questions. Overlap keeps a latency and energy advantage (half of each).
- **H6, overlap cuts escalation latency.** On escalated questions, `overlap` has lower median wall time than `freewidth`, because the big model starts during the first wave instead of after two failed waves. *Refuted if* escalated-question latency does not drop.
- **H7, the veto rescues confident-wrong answers.** `overlap` is more accurate than `freewidth` on questions the small model answers with a quorum, at mean time within 15% of `freewidth`. *Refuted if* the accuracy gain is zero or the time cost is larger.
- **H8, read-locate-fork replaces writing.** On questions where the small model is unsure, `repair` reaches at least the accuracy of `freewidth` (which escalates to big-model decoding) with lower mean latency, and the big model writes a full answer on fewer than half as many questions. *Refuted if* repaired batches are no more accurate than fresh small-model batches.
- **H9, the big model is better as a reader than a writer.** `freewidth`, `overlap` or `repair` beats `big_sc` (the 4B model alone) on accuracy per second **and** accuracy per joule on the T4. *Refuted if* `big_sc` dominates.
- **H5, wide beats long below the free width.** At equal device seconds, *w* ≤ free-width parallel non-thinking samples beat one thinking chain with a matching token budget.

## Method

### Three cost models (`freewidth/costs.py`)

For a sample with prompt length *P* and *L* generated tokens, using a model with active parameters *N*, *ℓ* full-attention layers, *h* heads and head dimension *d*:

- **FLOPs:** `2·N·L + 4·ℓ·h·d·Σ_t (P+t)`
- **eFLOPs (multi-tenant):** FLOPs `+ I · bytes(KV + recurrent state)`. *I* is the reference accelerator's arithmetic intensity (default 295 FLOP/byte for H100 BF16). Weight reads are excluded because they are amortised across tenants.
- **Device (single tenant, measured):** a wave of width *w* costs `T(w, L, P) = e + d·w·P + a·L + b·w·L + g·w·(L·P + L(L+1)/2)`. This is fitted with non-negative least squares on microbenchmarks. Energy (joules, via NVML) is fitted the same way. A verification pass is priced as `T(c, 1, P_v)`: one forward pass over *c* candidates plus one decoded token. **Free width** is the widest wave costing at most (1+ε) times one sample.

### Pipeline

1. `profile`: times forced-length decodes over widths × generation lengths (including L = 1, which anchors prefill) × at least two prompt lengths, then fits `T` and the free width. Check that R² > 0.98.
2. `bank`: draws *K* samples per (model, question) in waves and stores answer, correctness, tokens, self-certainty, entropy and text. Resumable. Thinking-mode tags use budget forcing: when the budget runs out, the thinking block is closed and an answer is forced.
3. `verify`: for each (small, big) pair, the big model scores every stored small-model solution with P(Yes | "is the final answer correct?"). This is one batched forward pass with no decoding.
4. `analyze`: replays every strategy across many random sample orders, prices each trace under every cost model, and writes `results.csv`, `disagreement.json` (tau, Jaccard, budget winners, overruns, saturating fits), `figures/` and `REPORT.md`.
5. `compare`: places several runs side by side (hybrid vs. dense, T4 vs. Mac) for H2.

Device cost always comes from the profile fit, never from the bank's own wall-clock. That means a bank generated with vLLM can be priced with an HF profile.

### Strategies (`freewidth/strategies.py`)

| family | type | description |
|---|---|---|
| `single` | baseline | one sample |
| `sc` | baseline | *n* parallel samples, majority vote (Wang et al. 2023) |
| `sc_seq` | baseline | the same *n* samples, one per round (identical FLOPs, different device cost) |
| `wvote` | baseline | Borda vote weighted by self-certainty (Kang et al. 2025) |
| `adaptive` | baseline | one sample at a time, Beta stopping (Aggarwal et al. 2023) |
| `short_mk` | baseline | *k* parallel; stop all rows when the first *m* finish (Hassid et al. 2025) |
| `conf_filter` | baseline | keep the most confident fraction, confidence-weighted vote (DeepConf offline style; mean-confidence variant) |
| `cascade` | baseline | vote on the small model; escalate if agreement < τ (FrugalGPT style) |
| `proxy_alloc` | baseline | dataset-level budget split by first-wave disagreement (DIPA style) |
| **`hwave`** | proposed | waves sized to the device's free width; Beta stopping after each wave; escalate a size only after waves fail |
| **`hwave(quorum=True)`** | proposed | **QuorumStop** inside each wave: stop the whole wave once the rows finished so far form a confident quorum; unfinished rows are cut and priced at the stop length |
| **`repair`** (engine) | proposed | Read-Locate-Fork: one big-model pass gives verdict + first flawed token; one-token correction; free-width small-model repair batch |
| `big_sc` (engine) | baseline | vote over samples of the largest model alone, one device |
| `big_sc2` (engine) | baseline | largest model data-parallel over all devices, one shared QuorumStop: the fair "big model on every GPU" reference |
| **`overlap`** (engine) | proposed | two-device ladder: streaming prefill verification with veto and verified accept, speculative big-model escalation, straggler cut |
| **`pverify`** | proposed | **Prefill-Verified Escalation**: free-width wave on the small model; if it is not confident, the big model prefill-scores the top distinct candidates; pick by P(Yes) (`max`) or vote share + P(Yes) (`mix`) |

## Running it

```bash
pip install -r requirements-dev.txt
make lint test                  # ruff + tests (the HF tests run where torch is installed)
python -m freewidth smoke       # full research pipeline on a simulated device, no GPU
make serve-mock                 # live engine + server on a simulated device
```

The smoke device is synthetic. Its numbers only prove the code runs; they are not results.

**Kaggle (GPU T4 x2, free).** Turn internet on, then run `bash scripts/kaggle_run.sh`. The script:

1. Profiles both families, hybrid on GPU 0 and dense on GPU 1.
2. Banks both families.
3. Runs the verification passes.
4. Analyses both runs.
5. Compares them in `runs/compare.md`.
6. Saves everything to `/kaggle/working/freewidth_runs.tgz`.

Re-running the script resumes unfinished work. Thinking-budget variants (`configs/kaggle_t4_think.yaml`) are optional and slow; they write into the hybrid run directory, so the next `analyze` includes them.

**Live engine on Kaggle:** run `bash scripts/kaggle_engine.sh`. It calibrates both GPUs, answers one question, and runs a live 100-question GSM8K benchmark of `single`, `sc`, `hwave` and `freewidth`, saving the results to `/kaggle/working/freewidth_engine.tgz`.

**Statistics for any run:** `python -m freewidth report <run dir> [<run dir> ...] --out <dir> --primary a:b,c:d`. This writes `REPORT.md`, `report.json` and a figure:
- accuracy with Wilson intervals; mean, median and p90 latency; energy; how often the big model decoded;
- every pair of policies: exact McNemar test, Holm correction within the pre-declared primary family, paired-bootstrap intervals, and the sample size needed for 80% power.

**Pre-registered experiments:** `bash scripts/kaggle_experiment.sh` with `CONFIG`, `DATASET`, `POLICIES` and `PRIMARY` set as in [docs/experiments.md](docs/experiments.md). It covers MATH-500, a second model pair, the P100 and the dense-model control.

**Faster banks:** `bash scripts/serve_vllm.sh Qwen/Qwen3-0.6B 0`, then use `configs/kaggle_t4_vllm.yaml`.
**Mac (Apple Silicon):** `configs/mac_mps.yaml`, a third hardware regime.

## Production engineering

- **Validated configs.** Both pipelines use pydantic schemas, so a typo fails at start-up rather than hours into a run.
- **Structured logging.** `--log-json` writes one JSON line per event, with a request id shared between the HTTP access log and the solve log.
- **Server.** OpenAI-compatible, with API-key auth, a bounded queue (`503`), request deadlines (`504`) and Prometheus `/metrics`.
- **CI and packaging.**
  - GitHub Actions run ruff, tests on Python 3.10–3.12, config validation, the smoke pipeline, and the real-transformers tests on CPU.
  - Actions are pinned to commit SHAs, with read-only permissions.
  - Also included: a CUDA Dockerfile (non-root, health check), a Makefile, a typed package (`py.typed`) and a changelog.
- **Security.** Details and the threat model are in [SECURITY.md](SECURITY.md).
  - **Server:** loopback by default, and it refuses an unauthenticated non-loopback bind. API key compared in constant time. Caps on body size, prompt, samples, waves and tokens. Public liveness probe only. No docs or internal error details exposed.
  - **Models:** loaded without `trust_remote_code`, with optional pinning to a Hub revision.
  - **Remote backends:** never send a key in clear text.
  - **CI scanning:** bandit, pip-audit, detect-secrets and CodeQL on every push and weekly. Dependabot is enabled.
- **Contributing:** [CONTRIBUTING.md](CONTRIBUTING.md).

Architecture overview: [docs/architecture.md](docs/architecture.md).

## Experiment plan

Every experiment's primary comparisons, sample size and pass/fail criteria are fixed in [docs/experiments.md](docs/experiments.md) before it runs:
- E1: fair two-GPU baseline and the repair retest;
- E2: MATH-500;
- E3: a second model pair;
- E4: the P100;
- E5: the dense-model control.

## Threats to validity

Full list with what the repo does about each: [docs/limitations.md](docs/limitations.md).


- **Engine:** HF `generate` is not a production server. Compare it with a vLLM profile where possible.
- **Hybrid kernels:** Gated DeltaNet layers need optimised kernels. The fallback path on a T4 may inflate device cost; record which kernels are installed.
- **Fit form:** `T` is linear in its features. Inspect residuals at large widths.
- **Verifier prompt:** P(Yes) depends on the prompt and on how the tokenizer splits Yes/No. Report the template, and check calibration with the `small_correct` field stored in each verify file.
- **Grading:** answer matching is rule-based. Spot-check disagreements using the stored texts.
- **Finite bank:** strategies that need more samples than the bank holds are truncated. Keep `samples_per_question` at least the largest *n* in the grid.

## References

- Snell, Lee, Xu, Kumar. *Scaling LLM Test-Time Compute Optimally Can Be More Effective than Scaling Model Parameters.* ICLR 2025. arXiv:2408.03314
- Wu et al. *Inference Scaling Laws.* arXiv:2408.00724
- Sadhukhan et al. *Kinetics: Rethinking Test-Time Scaling Laws.* NeurIPS 2025. arXiv:2506.05333
- Singhi et al. *When To Solve, When To Verify.* COLM 2025. arXiv:2504.01005
- Huang et al. *UniScale: Adaptive Unified Inference Scaling.* ICML 2026. arXiv:2605.30898
- *Resample or Reroute? Budget-Aware Test-Time Model Selection.* arXiv:2607.08665
- *Test-Time Scaling Makes Overtraining Compute-Optimal.* arXiv:2604.01411
- Hu, Shu, Ng, Low. *DIPA.* ICML 2026
- Zuo, Zhu. *Strategic Scaling of Test-Time Compute: A Bandit Learning Approach.* arXiv:2506.12721
- Hassid, Synnaeve, Adi, Schwartz. *Don't Overthink It: Preferring Shorter Thinking Chains.* arXiv:2505.17813
- Fu, Wang, Zhang, Tian, Zhao. *Deep Think with Confidence.* ICLR 2026. arXiv:2508.15260
- Aggarwal et al. *Let's Sample Step by Step: Adaptive-Consistency.* EMNLP 2023
- Muennighoff et al. *s1: Simple Test-Time Scaling.* 2025
- *SplitReason: Learning to Offload Reasoning.* arXiv:2504.16379
- Zhang et al. *Generative Verifiers: Reward Modeling as Next-Token Prediction.* 2024
- Wang et al. *Self-Consistency Improves Chain of Thought Reasoning.* ICLR 2023
- Kang et al. *Scalable Best-of-N Selection via Self-Certainty.* 2025
- Chen, Zaharia, Zou. *FrugalGPT.* 2023
- *Democratizing Agentic AI with Fast Test-Time Scaling on the Edge.* arXiv:2509.00195
- Narasimhan et al. *Faster Cascades via Speculative Decoding.* ICLR 2025
- *ConfSpec: confidence-gated cascaded verification for step-level speculative reasoning.* ACL 2026
- *Parallel Test-Time Scaling with Multi-Sequence Verifiers* (streaming MSV). arXiv:2603.03417
- *MARS: Margin-Adversarial Risk-controlled Stopping for Parallel LLM Test-time Scaling.* arXiv:2606.12935
- *Signed Rescue Routing: Harm-Aware Cascades for Efficient LLM Inference.* arXiv:2609.07786
- *Faster and Better LLMs via Latency-Aware Test-Time Scaling.* Findings of EMNLP 2025
- *StepBack: Step-Level Error-Localized Resampling for Efficient Test-Time Reasoning.* arXiv:2611.08403
- Shen, Sheng, Chen, Wang, Zhang. *Reasoning Can Be Restored by Correcting a Few Decision Tokens.* arXiv:2605.16874
- *Trajectory-Level Speculation with Semantic Verification for Tool-Using Agents* (SpecAgent). arXiv:2610.21463
- *DREAM-R: Multimodal Speculative Reasoning with Fully Parallel Execution.* arXiv:2605.28678
- Tsui. *Self-Correction Bench: Uncovering and Addressing the Self-Correction Blind Spot in Large Language Models.* arXiv:2507.02778

## License

MIT

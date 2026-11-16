# Limitations

The things a careful reviewer would raise, and what the repo does about each.

## Scope of the evidence
- **Small samples so far.** The live results use 100 questions per policy, which gives roughly ±8 points on accuracy. The 300-question run (E1) is sized from power estimates; claims wait for it. See `docs/experiments.md`.
- **One GPU type and one model family** (T4, Qwen3.5) so far. E3–E5 add a second model pair, the P100 and a dense model family. Until they run, the results describe one setting.
- **Grade-school math first.** GSM8K answers are numbers. MATH-500 (E2) tests harder, symbolic answers.

## Measurement
- **HF `generate`, not a serving engine.** Wall-clock times come from transformers' static-batch generation. A paged-attention server such as vLLM or SGLang has different batching costs, so free widths and timings may differ there.
- **Two models in one process.** This matters most for `big_sc2`. In E1 its two-GPU median (40 s) was slower than the one-GPU baseline's median from run 2 (31 s, different sample). The likely cause is interpreter-lock contention between the two decoding threads. A process-per-GPU version is planned (E1b).
  - `overlap` and `big_sc2` run GPU work from Python threads. Both GPUs are driven truly in parallel, but the Python-side overhead per decoding step is shared.
  - Before 0.5.1, every wave reseeded all GPUs with `torch.manual_seed`. In E1 the two GPUs' samples were nonetheless not duplicated ([details](results-t4-gsm8k-E1.md)). Since 0.5.1 each backend seeds only its own device.
  - Concurrent waves are still not bit-reproducible run to run. Per-question answers agreed on 96 of 100 questions across two runs.
- **Energy** is read from NVML for the GPUs a policy uses. An idle GPU's draw is not counted, which favours one-GPU policies on joules.
- **Verifier framing matters a great deal.** The same 4B model gave a median P(Yes) of 0.99 when the candidate was framed as its own answer, and 0.10 as a third-party answer. Every verifier result is specific to its prompt; templates are in `freewidth/engine/core.py` and `repair.py`.
- **Grading is automatic.**
  - GSM8K: exact number match after normalisation.
  - MATH-500: math-verify, with a string and number fallback.
  - Disagreements between policies can be spot-checked from the stored answers.

## Comparisons
- **Baselines.**
  - `big_sc` (runs 1–2) used one GPU and no early stopping. That was unfair to it on latency, and it is replaced by `big_sc2` from E1.
  - Comparisons with published methods (DeepConf, short-m@k and others) are offline replays on banked samples, not live reimplementations.
- **Multiple comparisons.** Each report corrects within its pre-declared primary family. Run 2's overlap vs freewidth result was p = 0.039 uncorrected and p = 0.077 after correcting for both primary comparisons, so it is reported as suggestive until E1.
- **Cross-session comparisons**, for example repair (E1) vs freewidth (run 2), are exploratory. GPU clocks, thermals and library versions can differ between Kaggle sessions.

## Novelty
- **Prior work.**
  - Latency-aware test-time scaling (EMNLP Findings 2025) already showed that compute-optimal is not latency-optimal.
  - Prefill verification, speculative cascades and early-stopped self-consistency all have prior work, cited in the README.
- **What is new here is narrower:**
  - free width measured per device and per model on single-tenant hardware, and used live;
  - QuorumStop inside a lock-step wave;
  - streaming verification with veto across two GPUs;
  - the Read-Locate-Fork repair loop.
- Each is tested against a fair baseline before it is claimed.

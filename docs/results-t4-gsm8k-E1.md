# E1: fair two-GPU baseline, Kaggle T4 x2, GSM8K, 300 questions (16–17 Nov 2026)

**Status: partial.**
- **Done:** `overlap` and `big_sc2` finished all 300 questions.
- **Unfinished:** the 12-hour Kaggle commit limit came before `repair`, the last policy, finished.
- **Source of these numbers:** the per-policy summaries in the run log. Paired tests need the per-question rows (`bench_rows.jsonl`) and are pending.
- **Plan:** pre-registered in [experiments.md](experiments.md#e1-fair-baseline-and-the-repair-retest).

**Setup**
- FreeWidth 0.5.0, GSM8K test, 300 questions. Sampler seed 0, so the 300 include runs 1–2's 100.
- Qwen3.5-0.8B on GPU 0 and Qwen3.5-4B on GPU 1, fp16, transformers with flash-linear-attention.
- Calibration (same session): 0.8B free width 16 (R² 0.998), 4B free width 4 (R² 0.997). This matches runs 1 and 2.
- `big_sc2` loaded a second 4B copy on GPU 0 next to the 0.8B model (8.2 s, no memory problems). It split 8 samples 4 + 4 across the GPUs, with one shared QuorumStop.

## Results (same session, same 300 questions)

| policy | accuracy (95% Wilson CI) | mean s | median s | p90 s | mean J | mean tokens | wall time |
|---|---|---|---|---|---|---|---|
| overlap | 89.3% (85.3–92.3) | **34.5** | **19.3** | 73.9 | **2,849** | 7,380 | 172 min |
| big_sc2 | **95.7%** (92.7–97.5) | 43.4 | 40.2 | **62.0** | 5,713 | 3,200 | 217 min |

**Overlap breakdown.**
- On 63% of questions, the 4B verified the 0.8B's answer without writing one.
- On 34%, the 4B started writing its own answer, including speculative starts that were later cancelled.
- On 27%, the answer was escalated to the 4B.

**Repeatability.** On the 300 questions, overlap scored 89.3% with a 19.3 s median. On run 2's 100 it scored 88.0% with an 18.7 s median. Its behaviour is stable across sample size and session.

## Pre-registered criteria

**E1.1, overlap vs big_sc2: the claim fails.**
- Overlap is faster: median half (19.3 vs 40.2 s), mean 8.9 s lower, energy half.
- Overlap is less accurate, by 6.4 points. The criterion required "not significantly less accurate".
  - The two accuracies differ by 19 questions.
  - If overlap gets 15 or fewer questions right that big_sc2 gets wrong, the exact McNemar p is below 0.01 (below 0.02 after Holm).
  - The exact value is pending the rows.

**Honest reading: this is a trade-off, not a win.** Overlap answers the typical question in half the time, for about 6 points less accuracy on GSM8K. Where accuracy matters most, run the 4B model on both GPUs.

**E1.2, repair vs overlap: not evaluated.** The run ended before repair completed.

## A caveat that makes big_sc2 look slower than it should

`big_sc2` was expected to be faster than the one-GPU `big_sc` from run 2. Its median was 31 s on a different 100-question sample, and the 4 + 4 split keeps each GPU at the 4B's free width. Instead the median rose to 40 s.

**Most likely cause: thread contention.** Both GPUs are driven by Python threads in one process. At small batch sizes, HF `generate` spends much of each decoding step in Python launching kernels, and the two threads take turns on the interpreter lock.

**Effect on the comparison.** If this is the cause, a one-process-per-GPU `big_sc2` would be faster still. That would *strengthen* the conclusion above, not weaken it. The overlap policy also uses threads, but its 4B mostly does single forward passes (verification), which are less sensitive.

**Next:** a process-per-GPU `big_sc2`, plus the one-GPU `big_sc` in the same session as a control.

## A bug found after the run: big_sc2's two GPUs probably drew the same samples

The code review for 0.5.1 found a seeding bug, now fixed:
- Each wave called `torch.manual_seed`, which reseeds **every** GPU in the process.
- `big_sc2` starts its two waves within milliseconds, so the second call most likely overwrote the first GPU's generator before either GPU sampled a token.
- With identical weights and prompts on two identical T4s, the two GPUs would then generate the same 4 samples.

**Consequences for E1:**
- `big_sc2` was most likely **4 distinct samples, each counted twice**, not 8 independent samples.
- Its accuracy (95.7%) is therefore a 4-sample result. The shared QuorumStop saw doubled votes.
- The comparison is still a fair "4B on two GPUs as built". But the baseline was weaker in samples than intended, which makes the 4B's accuracy lead over overlap *more* notable, not less.

**How we'll check:** the per-question rows store each wave's answers, so the duplication can be confirmed from `bench_rows.jsonl`.

**Fix (0.5.1):** each backend now seeds only its own GPU's generator. The overlap policy had the same process-wide reseed, but there the two GPUs run different models, so samples could not be duplicated. Only reproducibility was affected.

## Also learned

- **The run was too long.** About 8 h of estimated work in one 12 h commit, with the slowest policy last, left no margin. The fixed verifier also made repair slower than its 0.4.0 timing (it now rejects candidates, so it forks and falls back more).
- **From 0.5.0 onward, `bench --max-hours` stops starting new questions when the budget is spent.** It still writes every finished row and the report. The Kaggle scripts set the budget from the session clock, at 11.2 h total.

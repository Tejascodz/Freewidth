# Pre-registered experiments

Each experiment's primary comparisons, sample size and pass/fail criteria are written here **before** it runs. This file is version-controlled, so the commit history shows that the plan came first.

**Rules**
- **Primary comparisons are the only basis for a claim.** Holm correction is applied within the primary family (`freewidth report --primary ...`).
- **Everything else is exploratory.** It is corrected as its own family and reported as a lead, not a finding.
- **A failed criterion is reported as a failure.** It is not reworded or moved to a different subset.
- **Sample sizes come from the power estimates** in `freewidth report` (80% power, α = 0.05, exact McNemar) using the previous run's discordant rates.
- **All runs are free:** Kaggle T4 x2 or P100, open-weight models.

## Status

| id | question | setting | status |
|---|---|---|---|
| E0 | first live results | T4 x2, GSM8K, 100 q | done (runs 1–2): `docs/results-t4-gsm8k*.md` |
| E1 | does overlap still win against a fair two-GPU big-model baseline; does repair work once its verifier is fixed | T4 x2, GSM8K, 300 q | **partial**: E1.1 failed (trade-off, not a win); E1.2 not run (time limit). See `docs/results-t4-gsm8k-E1.md` |
| E1b | E1.1 rerun with independent per-GPU seeding (0.5.1), a process-per-GPU big_sc2, and a same-session one-GPU control; E1.2 repair in its own commit | T4 x2, GSM8K, 300 q | planned |
| E2 | does it hold on harder problems | T4 x2, MATH-500, 200 q | planned |
| E3 | does it hold for another model pair | T4 x2, Qwen3.5-2B → 4B, GSM8K, 300 q | planned |
| E4 | does the strategy ranking change with the hardware | P100 (one GPU), GSM8K, 300 q | planned |
| E5 | do hybrid (linear-attention) models change the small-vs-big trade-off (H2) | T4 x2, dense Qwen3 0.6B → 4B, GSM8K, 300 q | planned |

## E1: fair baseline and the repair retest

**Run.** `bash scripts/kaggle_final.sh`. Policies `overlap,big_sc2,repair` on GSM8K test, 300 questions (sampler seed 0; a superset of the 100 used before), Qwen3.5-0.8B on GPU 0 and Qwen3.5-4B on GPU 1. `big_sc2` runs the 4B on both GPUs: 8 samples split 4 + 4 (the 4B's free width), with a shared QuorumStop.

**Why 300.** From run 2's discordant rates:
- overlap vs the 4B alone needs about 280 questions;
- overlap vs freewidth needs about 145.

**Primary comparisons (Holm over these two)**
1. **overlap vs big_sc2**
   - *Claim if:* overlap's mean seconds are lower with a 95% interval below 0, and its accuracy is not significantly lower (Holm p ≥ 0.05, or overlap higher).
   - *Fail if:* big_sc2 is as fast or faster (interval includes 0 or favours big_sc2). Then the headline becomes "on two GPUs, run the big model on both", and overlap's speed claim is withdrawn.
2. **repair vs overlap**
   - *Claim if:* repair is not significantly less accurate than overlap (Holm p ≥ 0.05, or repair higher) **and** the largest model decodes on fewer questions.
   - *Fail if:* repair is significantly less accurate. Then H8 is refuted in this setting.

**Secondary, exploratory**
- **H8 as originally stated**, repair vs `freewidth`: on the 100 questions shared with run 2. These are cross-session, so the result is exploratory.
- **Overlap repeatability:** overlap run 2 vs overlap E1, on the shared 100 questions.
- **Verifier calibration under the 0.4.1 framing:** the distribution of repair's P(Yes) on correct vs incorrect candidates.
- **How often big_sc2's shared QuorumStop fires**, and its effect on latency.

## E2: MATH-500

**Run.**
```bash
CONFIG=configs/engine_t4_math.yaml DATASET=math500 LIMIT=200 \
  POLICIES=overlap,big_sc2 PRIMARY=overlap:big_sc2 bash scripts/kaggle_experiment.sh
```
Answers are graded with math-verify. `max_new_tokens` is 2048, with its own calibration, because free width depends on the generation length.

**Primary.** overlap vs big_sc2, with the same criteria as E1.1.

**Secondary.** Results by MATH level (1–5). We expect the gap to grow with difficulty, but this is exploratory.

**Size.** 200 questions keeps the run under 12 h. If E1 shows an accuracy gap smaller than 5 points, E2 is reported as a latency comparison with an accuracy confidence interval, not an accuracy claim.

## E3: second model pair

**Run.**
```bash
CONFIG=configs/engine_t4_2b.yaml POLICIES=overlap,big_sc2,freewidth \
  PRIMARY=overlap:big_sc2,overlap:freewidth bash scripts/kaggle_experiment.sh
```
**Primary.** As in E1.1, plus overlap vs freewidth (overlap more accurate, Holm p < 0.05).

**Note.** The 2B and the replica 4B on GPU 0 need about 14 GB. If the replica does not fit, `big_sc2` runs on one GPU and the run notes say so. In that case the fair-baseline comparison is **not** claimed for this pair.

## E4: different hardware (P100)

**Run.**
```bash
CONFIG=configs/engine_p100.yaml POLICIES=freewidth,big_sc,sc \
  PRIMARY=freewidth:big_sc bash scripts/kaggle_experiment.sh
```
**Claim (hardware-conditional ranking).**
- The measured free widths differ from the T4's.
- The ordering of `freewidth` vs `big_sc` by mean seconds at matched accuracy differs from the T4 ordering.

*Fail if* the free widths and the ordering both match the T4. In that case the paper does not claim that the ranking depends on hardware.

**Risk.** Qwen3.5's linear-attention kernels may not run on the P100's sm_60 architecture. If calibration falls back to the slow path, switch to the dense pair as described in the config header, and report it.

## E5: dense control (H2)

**Run.**
```bash
CONFIG=configs/engine_t4_dense.yaml POLICIES=overlap,big_sc2,freewidth \
  PRIMARY=overlap:big_sc2 bash scripts/kaggle_experiment.sh
```
**Claim.** Free widths and the overlap vs big-model trade-off differ between hybrid Qwen3.5 and dense Qwen3 at similar sizes. This is reported descriptively, since model families also differ in accuracy.

## Budget

| id | approx. GPU hours |
|---|---|
| E1 | 8 |
| E2 | 8–10 |
| E3 | 8 |
| E4 | 6 |
| E5 | 8 |

Kaggle gives about 30 GPU hours a week, so the plan takes about two weeks of free quota.

# Live engine and server

The research pipeline (`profile → bank → verify → analyze`) studies strategies offline. The engine runs the same ideas **live**, on a ladder of local models, request by request.

## How a request is answered (policy `freewidth`)

```
question
  │
  ▼
small model: wave of w = free width samples ──► QuorumStop inside the wave
  │   rows finish at different times; once the finished rows agree
  │   (Beta posterior ≥ tau) the wave stops and still-running rows are cut
  │
  ├─ confident? ──► answer (shortest chain that supports the winning answer)
  │
  ├─ up to max_waves waves, votes pooled across waves
  │
  ▼ still unsure
big model: ONE forward pass over the top candidates, P(Yes | "is this correct?")
  │
  ├─ best candidate (vote share + P(Yes)) has P(Yes) ≥ verify_accept ──► answer
  │
  ▼ rejected
big model decodes its own waves (same QuorumStop) ──► answer
```

### Policy `overlap` (two-device ladder, proposed in 0.3.0)

`freewidth` is a sequential cascade: the big model only works after the small model has given up. In the T4 run, that left GPU 1 idle 90% of the time, and escalated questions spent about 99 s on the small model before escalating. `overlap` puts the idle GPU to work while the small model is still decoding:

```
GPU 0 (small)  ── wave of free width ─────────────┬── answer (big model run cancelled)
                 │ row finishes ─► queue           │
GPU 1 (big)      ├─► prefill P(Yes) per row ───────┘  streaming verification
                 └─► failure signal ─► big model starts decoding NOW (speculative escalation)
```

- **Streaming verification.** Each finished small-model row is scored by the big model at once, overlapping with the remaining decode.
- **Veto.** A confident small-model majority is not accepted if the big model scores all of its chains below `veto_lo`. That targets confident-but-wrong answers: 17 of the 82 quorum answers in the T4 run were wrong.
- **Verified accept.** An answer with at least 2 votes and a big-model score ≥ `accept_hi` (0.95) is accepted even without a quorum. In the T4 run, the wrong answers the verifier accepted all scored between 0.78 and 0.91.
- **Speculative escalation.** The big model starts decoding as soon as the wave signals failure: the finished rows disagree (`trigger_lead_prob`), the lead answer is vetoed, or stragglers outlive the finished rows by `runaway_factor`. If the small model then accepts, the big model's run is cancelled.
- **Straggler cut.** Rows still running beyond `runaway_factor` × the median finished length stop and do not vote.

It needs two ladder models, ideally on different GPUs (the default T4 config). On a single GPU it still runs, but the two models compete for the device. `wall_seconds` reports the overlapped end-to-end time, and `notes` records the trigger, the accept reason, verified rows and cut rows.

### Policy `repair` (Read-Locate-Fork, proposed in 0.4.0)

The big model reads once and writes at most one token:

1. **Read.** The small model's candidate is placed as the big model's own answer to the question, followed by "Is the final answer of your solution above correct?". One forward pass gives the verdict, P(Yes), and the big model's next-token distribution at every solution token.
2. **Locate.** The flaw is the first solution token (after a short warm-up) that the big model rates below `flaw_p` (0.05) while preferring another token with at least `flaw_conf` (0.5).
3. **Fork.** The big model's token replaces the small model's at the flaw. The small model continues from there in a full free-width batch with QuorumStop, and the samples vote.
4. **Accept or escalate.** A verified candidate (P(Yes) ≥ `accept_hi`) is accepted without repair. A repaired answer is accepted if its batch reaches a quorum and the big model does not veto it. Otherwise the big model decodes as a last resort.

It needs both models to share a tokenizer (one model family); otherwise it falls back to verification and escalation. `notes` records `repaired_from`, `flaw_token`, `repair_failed` and the accept reason.

**Closest prior work:**
- **StepBack (2026):** first-error localization plus suffix resampling, using the same model or an external scorer.
- **Shen et al. 2026:** a few decision tokens carry the gap between a base model and a reasoning model.
- **SpecAgent (2026):** the strong model prefill-verifies a weak draft and repairs the first rejected step itself.

What's new here: the verdict and the location come from the **same pass**, the strong model writes **one token**, and the repair is a **parallel free-width batch** of the weak model.

### Baselines `big_sc` and `big_sc2`

`big_sc`: `sc_n` samples from the **largest** ladder model in one batch on its own device, with a majority vote. Other devices stay idle.

`big_sc2` is the fair "just run the big model on all the GPUs" baseline. The largest model is also loaded on every device in its `replica_devices` list, the first time the policy is used:

```yaml
- {tag: q35-4b, backend: hf, model_id: Qwen/Qwen3.5-4B, device: "cuda:1", dtype: float16,
   replica_devices: ["cuda:0"]}
```

- **Data-parallel waves.** The `sc_n` samples are split evenly over the copies (8 over 2 GPUs = 4 each, the 4B's free width on a T4), and all waves run at the same time.
- **One shared QuorumStop.** The waves pool their votes in a `SharedQuorum`, so every device stops as soon as the pooled votes are confident. `quorum: false` turns this off.
- **Degrades instead of failing.**
  - A replica that fails to load (for example, out of memory next to the small model) is skipped. The policy runs on the devices that did load, and the error appears in `/v1/info` and in the request's `notes`.
  - A replica that runs out of memory mid-request has its share rerun on the primary.
- **Timing.** The reported time is the real wall-clock of the parallel waves. Energy is the sum over all devices used.

Other policies, chosen per request with `freewidth.name`:

| policy | behaviour |
|---|---|
| `single` | one sample from the first model |
| `sc` | `sc_n` parallel samples, majority vote |
| `hwave` | free-width waves with QuorumStop, escalate by decoding (no verification) |
| `freewidth` | `hwave` plus prefill verification before every escalation (default) |
| `overlap` | two-device ladder: streaming verification, veto, verified accept, speculative escalation, straggler cut |
| `repair` | Read-Locate-Fork: one big-model pass gives verdict + flaw; one-token correction; free-width small-model repair batch |
| `big_sc` | baseline: vote over `sc_n` samples of the largest model, one device |
| `big_sc2` | baseline: largest model on every device holding a copy, samples split evenly, one shared QuorumStop |

**Free width** is measured, not guessed. `calibrate` times forced-length decodes at several widths, fits the cost curve `T(w, L, P)`, and picks the widest wave that costs at most (1 + `eps`) times a single sample. The result is stored per model and per device in `calibration/<tag>/device_fit.json`. If the device name changes, the stored calibration is ignored and measured again.

## Commands

```bash
python -m freewidth calibrate --config configs/engine_t4.yaml          # one-time per device
python -m freewidth solve     --config configs/engine_t4.yaml "question" [--policy hwave] [--json]
python -m freewidth serve     --config configs/engine_t4.yaml [--port 8000] [--log-json]
python -m freewidth bench     --config configs/engine_t4.yaml --dataset gsm8k --limit 100 \
                              --policies single,sc,hwave,freewidth --out runs/bench
```

`bench` measures every number live (wall seconds, NVML joules, tokens) and writes `BENCH.md`, `bench_summary.csv` and per-question `bench_rows.jsonl`. Use it to check the offline replay's predictions on real hardware.

## HTTP API

`POST /v1/chat/completions` is OpenAI-compatible:

```json
{
  "model": "freewidth",
  "messages": [{"role": "user", "content": "What is 17 * 23?"}],
  "temperature": 0.7,
  "max_tokens": 1024,
  "seed": 1,
  "freewidth": {"name": "freewidth", "tau": 0.9, "max_waves": 2, "verify": true}
}
```

The response has the standard `choices` and `usage`, plus a `freewidth` object:

```json
"freewidth": {
  "request_id": "…", "policy": "freewidth", "answer": "391", "confidence": 0.94,
  "model": "q35-0.8b", "escalated": false, "verified": false, "free_form": false,
  "waves": [{"model": "q35-0.8b", "width": 16, "finished": 9, "cut": 7, "tokens": 3120,
             "seconds": 4.1, "joules": 290.5, "quorum_hit": true}],
  "verifications": [], "tokens": 3120, "seconds": 4.1, "joules": 290.5
}
```

`usage.completion_tokens` counts tokens across **all** samples, because that is the real compute spent.

Allowed `freewidth` keys: `name`, `tau`, `max_waves`, `quorum`, `min_finished`, `width`, `max_width`, `verify`, `verify_accept`, `top_c`, `sc_n`, `free_form_fallback`, `raw_prompt`, and for `overlap`: `veto_lo`, `accept_hi`, `require_verified`, `stream_verify`, `speculative`, `trigger_min_finished`, `trigger_lead_prob`, `runaway_factor`, `runaway_min_finished`, `verify_wait_s`, and for `repair`: `flaw_p`, `flaw_conf`. Values are validated, and server-side caps apply on top (`server.max_samples` for `sc_n`/`width`/`max_width`, `server.max_waves`, `server.max_new_tokens`, `server.allowed_policies`); anything else returns 400.

Other endpoints:

| endpoint | purpose |
|---|---|
| `GET /health` | liveness only (`{"status": "ok"}`); public, for load balancers and container probes |
| `GET /v1/info` | uptime, queue depth, version, ladder with devices, free widths and calibration R² (needs the API key) |
| `GET /v1/models` | the engine name plus every ladder tag |
| `GET /metrics` | needs the API key unless `server.public_metrics`. Prometheus text: requests, waves, quorum stops, escalations, verifications, tokens, joules, latency histogram, queue depth, free width per model |

`stream: true` is accepted for client compatibility. The whole answer arrives as one SSE chunk, because the answer is only known after voting.

## Operations

- **Single tenant by design.** Requests run one at a time, because the engine owns the GPU and the free width assumes nothing else is running on it. Up to `server.max_queue` requests wait; beyond that the server returns `503` with `Retry-After`.
- **Deadlines.** Each request gets `server.request_timeout_s`. The engine checks the deadline before every wave and every verification pass and returns `504` once it is exceeded. A wave that has already started finishes first.
- **Auth.**
  - If the environment variable named by `server.api_key_env` (default `FREEWIDTH_API_KEY`) is set, every endpoint except `/health` needs `Authorization: Bearer <key>`. The key is compared in constant time.
  - The server binds to `127.0.0.1` by default. On any other address it refuses to start without a key, unless `server.allow_unauthenticated: true` or `--insecure-no-auth` is set.
- **Request limits.** These bound how much GPU time and memory one request can claim:
  - body size (`max_body_bytes`; POST requests must send `Content-Length`);
  - message count (`max_messages`) and prompt length (`max_prompt_chars`);
  - samples, waves and new tokens, as above.
- **Other hardening.**
  - `/docs` and `/openapi.json` are off unless `server.expose_docs` is set.
  - Responses carry `X-Content-Type-Options: nosniff` and `Cache-Control: no-store`.
  - Client-supplied request ids are accepted only if they match `[A-Za-z0-9._-]{1,64}`.
  - Errors return no internal details.
  - See [SECURITY.md](../SECURITY.md).
- **Logs.** `--log-json` writes one JSON object per line. Every HTTP request and its solve share a `request_id`, which is also returned in the `x-request-id` header.
- **Docker.**
  - Build and run: `docker build -t freewidth . && docker run --gpus all -p 127.0.0.1:8000:8000 -e FREEWIDTH_API_KEY=<key> -v $PWD/calibration:/app/calibration freewidth`.
  - The image runs as a non-root user and listens on all interfaces inside the container, so it needs the key.
  - Mount `calibration/` so calibration survives restarts. Mount the Hugging Face cache at `/home/app/.cache/huggingface` to avoid re-downloading models.
- **Model provenance.**
  - Models load with `trust_remote_code` off, so no Python from a model repository runs.
  - Set `revision:` on a ladder entry to pin the exact Hub commit. Without it, the engine logs a warning at load.
- **Memory.** The default T4 config puts the small model on `cuda:0` and the big model on `cuda:1`. On one 16 GB GPU, 0.8B + 4B in fp16 fits, but keep `max_width` at 16–32 for 1k-token generations.

## Limits

- **Checkable answers only.** The engine is built for questions with a checkable final answer (math, multiple choice, short factual answers): voting needs answers it can extract. For free-form prompts, `free_form_fallback` returns the most confident sample after the first wave and does not escalate.
- **No thinking-budget variants yet.** The research pipeline has them; the engine ladder does not.
- **Prompt-sensitive verification.** Prefill verification quality depends on the verifier prompt and model. Measure acceptance accuracy with `bench` before relying on it, and tune `verify_accept`.

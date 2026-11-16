# Security policy

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub: **Security → Report a vulnerability** on this repository. Do not open a public issue for a security problem.

Include:
- the affected version or commit;
- steps to reproduce;
- the impact as you understand it.

You can expect an acknowledgement within 7 days. Fixes for confirmed issues are released as a new version with a changelog entry.

## Supported versions

Only the latest release on `main` receives security fixes.

## Threat model

FreeWidth runs open-weight language models on GPUs the operator controls. It exposes an OpenAI-compatible HTTP server. The defaults assume the server is **not** on the public internet. Here is what the code defends against, and what is left to the operator.

| risk | defence in the code | operator's responsibility |
|---|---|---|
| Unauthenticated use of the GPU | Bearer API key, compared in constant time (`hmac.compare_digest`). Binds to `127.0.0.1` by default. Refuses to listen on another address without a key unless `--insecure-no-auth` / `server.allow_unauthenticated` is set. | Generate a long random key (`openssl rand -hex 32`), keep it in a secret store, and rotate it. |
| One request monopolising the GPU or memory | Caps on body size (Content-Length required), messages, prompt characters, samples (`sc_n`/`width`/`max_width`), waves and new tokens. Policy allow-list. Bounded queue (`503`). Per-request deadline (`504`). | Tune the limits in `server.*` for your hardware. Put a rate limiter in front for multi-user access. |
| Information disclosure | `/health` returns liveness only. Engine details (`/v1/info`) and `/metrics` need the key. OpenAPI docs are off by default. Errors return no internal details. No `Server` header. Prompts and answers are not logged. | Treat logs as sensitive anyway: request metadata is logged. |
| Log forging | Client request ids must match `[A-Za-z0-9._-]{1,64}`. Text logs escape line breaks. | Prefer `--log-json` in production. |
| Malicious model repositories | `trust_remote_code` is off: no Python from a model repo is executed. Optional `revision:` pin per model (a warning is logged when unpinned). Weights load through `transformers`, which uses safetensors where available. | Pin `revision` to a reviewed commit for production. Mirror the weights if supply-chain integrity matters. |
| Leaking API keys for remote models | The OpenAI-compatible backend accepts only `http(s)` URLs. It refuses to send a key over plain `http` to a non-loopback host unless `allow_insecure_http: true`. Keys are read from environment variables, never from config files. | Never put keys in configs or commits; `.gitignore` excludes `.env*`, `*.key`, `*.pem`, `kaggle.json`. |
| Unsafe deserialisation | YAML is read with `yaml.safe_load`. Calibration, results and reports are JSON. No `pickle`, `eval`, `exec` or shell commands. | Only load configs you trust. They choose which models run. |
| Container escape impact | The image runs as an unprivileged user (uid 10001). | Run with `--gpus` only, and publish the port on `127.0.0.1` unless remote access is needed. |
| Vulnerable dependencies, CI supply chain | CI runs bandit, pip-audit and detect-secrets on every push and weekly, plus CodeQL. GitHub Actions are pinned to commit SHAs and run with read-only permissions. Dependabot proposes updates. | Review and merge Dependabot PRs. |

## Out of scope

- **Prompt injection.** The engine answers questions. It has no tools, file access or network access that a prompt could misuse.
- **Model output quality.** Wrong answers are a research topic here, not a vulnerability.
- **Research pipeline scripts** (`profile`, `bank`, `analyze`, the Kaggle scripts). They run locally on files the operator controls.

# Contributing

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
make lint test
```

The HF tests run when `torch` and `transformers` are installed. GPU tests run only with `FREEWIDTH_GPU_TESTS=1`.

## Rules for changes

- **Every behaviour change comes with a test.** The test suite runs on a simulated device, so no GPU is needed.
- **Results are pre-registered.** New experiments go into the README's [pre-registered experiments](README.md#experiments) section *before* they run:
  - the primary comparisons;
  - the sample size, from the power estimate in `freewidth report`;
  - what counts as failure.
- **A failed criterion is reported as a failure.**
- **Numbers in docs come from `freewidth report`** run on committed result files, not from hand calculations.
- **Security-sensitive code** (the server, backends, anything that reads files or the network) is read against [SECURITY.md](SECURITY.md). New request parameters need a server-side cap.
- **Never commit keys or tokens.** CI runs a secret scanner.

## Pull requests

CI must pass:
- ruff;
- tests on Python 3.10–3.12;
- the real-transformers tests on CPU;
- config validation and the smoke pipeline;
- bandit, pip-audit, detect-secrets and CodeQL.

Describe what changed and how it was tested. For engine changes, also give the expected effect on accuracy or latency.

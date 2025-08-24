# LLDC — LLMs as Data Compressors

Reproducible research code for **Predictive Masking (PM)** and **Vector-Quantization (VQ)** compression pipelines.

## TL;DR

```bash
# 1) One-shot setup (Ubuntu/Debian or Arch)
chmod +x setup.sh && ./setup.sh

# 2) (Optional) Hugging Face token for gated models
cp .env.example .env  # add HUGGING_FACE_TOKEN=...
# Then either: export $(grep -v '^#' .env | xargs)
# Or add dotenv in code

# 3) Run the main experiment
make run-exp ARGS="+experiment=e1a_wiki103"
```

## What’s here

- `configs/` – Hydra configs for **compute**, **data**, **model**, **eval**, **experiment**
- `lldc/` – source (compression, decompression, metrics, models, utils, scripts)
- `artifacts/` – outputs: checkpoints, logs, runs, RD curves
- `tests/` – unit tests
- `external/` – optional third-party tools (cmix, etc.)
- `Makefile`, `pyproject.toml`, `uv.lock` – tooling & deps

## Requirements

- Linux (Ubuntu/Debian or Arch), Python **3.10+**
- `cmix` benefits from **≥32 GB RAM**
- Uses [`uv`](https://github.com/astral-sh/uv) for fast, reproducible environments

## Setup

`setup.sh` installs:

- System build deps + `zstd`
- **KenLM** binaries (`lmplz`, `build_binary`)
- **cmix** (g++)
- `uv` and project deps (incl. dev tools)
- Exports `KENLM_BIN` and updates `PATH`

Run:

```bash
./setup.sh
```

Quick sanity checks:

```bash
lmplz -h && build_binary -h && cmix
```

## Running experiments

Use the Make target (wraps Hydra sweeps):

```bash
# Generic
make run-exp ARGS="+experiment=<name> [overrides]"

# Common
make run-exp ARGS="+experiment=e1a_wiki103"
make run-exp ARGS="+experiment=e2a_pruning"
make run-exp ARGS="+experiment=e2b_channel"

# Change dataset/model/compute
make run-exp ARGS="+experiment=e1a_wiki103 data=wikitext-2 model=gpt2-medium compute=local_dev"
```

Hydra writes all outputs to `artifacts/`.

## CLI entry points

Registered in `pyproject.toml` (use `uv run`):

```bash
uv run specialise
uv run compress_pm
uv run compress_vq
uv run reconstruct_pm
uv run reconstruct_vq
uv run evaluate
uv run sweeps +experiment=e1a_wiki103
```

## Dev & QA

```bash
make fmt        # ruff fix + format
make lint       # ruff + mypy
make test       # pytest (quiet)
make test-verbose
make test-cov
make all        # sync + lint + test
```

## Where outputs go

- Checkpoints / logs / runs → `artifacts/`
- Rate–distortion curves → `artifacts/rd_curves`
- Reports (figures/tables) → `reports/`

## Troubleshooting (short)

- **KenLM not found**: `echo "$KENLM_BIN"` and ensure `lmplz` is on `PATH` (open a new shell or `source ~/.bashrc` / `~/.zshrc`).
- **bitsandbytes** issues (non-Linux): optional; you can remove it or install on Linux only.
- **cmix memory**: start with small files if <32 GB RAM.
- **.env not loaded**: export it (see TL;DR) or use `dotenv` in code.

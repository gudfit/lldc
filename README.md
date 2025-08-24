# LLDC — Part A (Thesis Code)

**LLMs as Data Compressors**: reproducible research code for **Predictive Masking (PM)** & **Vector-Quantization (VQ)** compression pipelines.

---

## What’s inside (at a glance)

* `configs/` — Hydra configs for **compute**, **data**, **model**, **eval**, **experiment** (e.g., `experiment/e1a_wiki103.yaml`, `data/wikitext-2.yaml`, `model/gpt2.yaml`).
* `lldc/` — source code (compression, decompression, metrics, models, scripts, utils).
* `artifacts/` — where **checkpoints**, **logs**, **runs**, **rd_curves** are written.
* `tests/` — unit tests for compression & metrics.
* `external/` — optional external tools (e.g., `cmix`, `deepzip`—but we'll install KenLM, zstd, cmix below).
* `Makefile`, `pyproject.toml`, `uv.lock` — tooling & deps.

---

## 1) One-time setup

This is the heavy lifting—installing Python tools, building C++ binaries for baselines, and setting paths. I assume you're on Ubuntu/Linux; if not, check the tool-specific notes. Budget 30-60 mins, especially if builds hit snags (like missing deps or APT locks).

### (A) Install uv (Python package manager)

uv is awesome for fast, reproducible envs. Install it globally:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Caveat: If you're behind a proxy or on a restricted system, download manually from https://github.com/astral-sh/uv. After install, close/reopen your terminal or source your shell config (e.g., `source ~/.bashrc`).

### (B) System prerequisites

These are build essentials and libs needed for compiling KenLM, cmix, etc. Run as root or with sudo:

```bash
sudo apt-get update && sudo apt-get install -y python3.10-dev build-essential libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev git cmake libboost-all-dev
```

Why these? Python dev for bindings, build-essential/cmake/git for compiling, Boost/Eigen/zlib/etc. for KenLM/cmix deps.

Caveat: If APT is locked (e.g., "Could not get lock /var/lib/dpkg/lock-frontend" from unattended-upgrades), wait 5-10 mins or check `ps aux | grep unattended`. If stuck, kill the process (e.g., `sudo kill <PID>`), remove locks (`sudo rm /var/lib/dpkg/lock*`), and fix with `sudo dpkg --configure -a && sudo apt install -f`. Logs in `/var/log/unattended-upgrades/` help debug.

### (C) Make your `.env`

For Hugging Face auth (needed for model downloads):

```bash
cp .env.example .env
```

Edit `.env` with your token:

```bash
HUGGING_FACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxx
```

Export it (run this in every new terminal, or add to `~/.bashrc`):

```bash
export $(grep -v '^#' .env | xargs)
```

Or POSIX-safe:

```bash
set -a
. ./.env
set +a
```

Caveat: If using Jupyter/VSCode, add `from dotenv import load_dotenv; load_dotenv()` in your code.

### (D) Install Python dependencies with uv

Requires Python 3.10+. This sets up a virtual env with project deps (including dev tools for linting/tests):

```bash
make dev
```

This uses `pyproject.toml` to create `.venv/` and install everything. Activate with `source .venv/bin/activate` if needed, but `uv run` handles it.

Caveat: If `bitsandbytes` fails (non-Linux), it's optional—ignore or comment out in `pyproject.toml`.

### (E) External dependencies

These are for baselines in `lldc/scripts/compute_baselines.py` and `channel_analysis.py`. They're C++ tools, so we build/install them manually.

#### 1. KenLM (N-gram language modeling for entropy baselines)

Why: Calculates N-gram entropy and regeneration baselines.

Build from source (no APT package for binaries):

```bash
git clone https://github.com/kpu/kenlm.git
cd kenlm
mkdir -p build
cd build
cmake ..
make -j4  # Adjust -j to your cores
sudo make install  # Optional, for /usr/local/bin
```

Set path (add to `~/.bashrc` and `source ~/.bashrc`):

```bash
export KENLM_BIN="$HOME/kenlm/build/bin"
```

Verify: `$KENLM_BIN/lmplz --help` should work.

Caveats:
- If CMake fails on Boost: `sudo apt install libboost-all-dev`. Rerun `cmake ..` after.
- Missing libs? Install `sudo apt install zlib1g-dev libbz2-dev liblzma-dev`.
- Test build: `echo "this is a test" > test.txt; $KENLM_BIN/lmplz -o 3 < test.txt > test.arpa; $KENLM_BIN/build_binary test.arpa test.binary`.
- macOS: Use Homebrew for deps (`brew install cmake boost zlib bzip2 xz git`).
- Windows: Use WSL or vcpkg.

#### 2. zstd (Zstandard compressor)

Why: Modern lossless compression baseline.

Simple install:

```bash
sudo apt update && sudo apt install zstd
```

Verify: `zstd --help`.

Caveats: On macOS: `brew install zstd`; Fedora: `sudo dnf install zstd`. It's in PATH after install.

#### 3. cmix (Context-mixing compressor)

Why: Top-tier lossless baseline for benchmarks.

Build from source (needs 32GB+ RAM to run effectively):

```bash
git clone https://github.com/byronknoll/cmix.git
cd cmix
```

Install Clang-17 (recommended):

If not in repos:

```bash
sudo apt update && sudo apt install software-properties-common wget lsb-release gnupg
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 17
sudo apt install clang-17
```

Compile:

```bash
make
```

Move to PATH:

```bash
sudo mv cmix /usr/local/bin/
```

Verify: `cmix` should show usage.

Caveats:
- If `clang-17` not found: Use the LLVM script above.
- Compilation errors (missing headers like `<fstream>`): `sudo apt install build-essential`.
- Use GCC alternative: Edit `makefile` (`nano makefile`), change `CC = clang++-17` to `CC = g++`, then `make clean && make`. Needs g++ 11+ (`g++ --version`).
- High RAM needed—test on small files first.
- If locks during install: Same as (B)—wait or kill unattended-upgrades.

---

## 2) Run experiments (the main workflow)

With setup done, this is easy. Use Make to run Hydra sweeps:

```make
run-exp:
	uv run python -m lldc.scripts.sweeps $(ARGS)
```

### Common runs

* **Experiment 1A (WikiText-103, main compression pipeline)**

  ```bash
  make run-exp ARGS="+experiment=e1a_wiki103"
  ```

* **Experiment 2A (Pruning)**

  ```bash
  make run-exp ARGS="+experiment=e2a_pruning"
  ```

* **Experiment 2B (Channel analysis)**

  ```bash
  make run-exp ARGS="+experiment=e2b_channel"
  ```

* **Change dataset (example: WikiText-2)**

  ```bash
  make run-exp ARGS="+experiment=e1a_wiki103 data=wikitext-2"
  ```

Hydra merges configs from `configs/defaults.yaml`. Outputs to `artifacts/` (checkpoints, logs, runs, RD curves).

Caveat: Add `+` before args if overriding deeply (Hydra quirk).

---

## 3) Useful CLI entry points

Registered in `pyproject.toml`, run with `uv run`:

```bash
# Stage 1: specialize models
uv run specialise

# Stage 2: compress with PM or VQ
uv run compress_pm
uv run compress_vq

# (If present) Stage 3: reconstruct
uv run reconstruct_pm
uv run reconstruct_vq

# Evaluate suite
uv run evaluate

# Full sweeps
uv run sweeps +experiment=e1a_wiki103
```

Prefer `make run-exp` for reproducibility.

---

## 4) Development & QA

* **Format + Lint**

  ```bash
  make fmt
  make lint
  ```

* **Run tests**

  ```bash
  make test  # quiet
  make test-verbose  # verbose
  make test-cov  # coverage
  ```

* **Everything (CI-ish)**

  ```bash
  make all  # sync + lint + test
  ```

---

## 5) Configuration quick tips

* **Compute**: `configs/compute/{local_dev.yaml,crescent_v100.yaml}`
* **Data**: `configs/data/{wikitext-2.yaml,wikitext-103.yaml,the-stack.yaml,mathematics.yaml}`
* **Model**: `configs/model/{gpt2.yaml,gpt2-medium.yaml,gpt2-large.yaml,bert-base-cased.yaml,roberta-base.yaml,distilroberta.yaml}`
* **Eval**: `configs/eval/{fidelity.yaml,efficiency.yaml,crumpled_paper.yaml}`

Override via CLI:

```bash
make run-exp ARGS="+experiment=e1a_wiki103 model=gpt2-medium data=wikitext-2 compute=local_dev"
```

---

## 6) Where things go

* **Checkpoints / logs / runs** → `artifacts/`
* **Rate–distortion curves** → `artifacts/rd_curves`
* **Figures & tables (reports)** → `reports/{figures,tables}`

---

## 7) Troubleshooting

* **.env not picked up**: Re-export or use `dotenv` in code.
* **bitsandbytes issues**: Optional on non-Linux.
* **KenLM not found**: Check `KENLM_BIN` points to `lmplz` dir. Re-source shell.
* **Build errors**: Missing deps? Rerun apt installs. Share CMake/make output.
* **APT locks/stuck upgrades**: Monitor logs, kill if needed, clean locks.
* **cmix RAM**: Needs 32GB+—downscale experiments if low.
* **General**: Ensure paths (e.g., `echo $PATH` includes /usr/local/bin). If stuck, check issues or add yours!

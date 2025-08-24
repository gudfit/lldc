#!/usr/bin/env bash

# setup.sh — one-shot bootstrap for LLDC (Ubuntu/Debian & Arch)
# - Installs uv, system deps, zstd
# - Builds & installs KenLM binaries (lmplz, build_binary)
# - Builds cmix with g++ (not clang-17), installs to /usr/local/bin
# - Sets PATH and KENLM_BIN
# - uv sync (incl. dev group) using your pyproject/makefile
# - Cleans up temp build dirs & apt cache

# Run from repo root:  chmod +x setup.sh && ./setup.sh

set -euo pipefail

log()  { printf "\n\033[1;36m[LLDC]\033[0m %s\n" "$*"; }
die()  { printf "\n\033[1;31m[LLDC ERROR]\033[0m %s\n" "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

SHELL_RC="$HOME/.bashrc"
if [[ "${SHELL:-}" == *zsh ]]; then SHELL_RC="$HOME/.zshrc"; fi

add_line_once() {
  local line="$1" file="$2"
  grep -qxF "$line" "$file" 2>/dev/null || echo "$line" >>"$file"
}

TMP_ROOT="$(mktemp -d -t lldc-setup-XXXXXX)"
cleanup() {
  log "Cleaning temp: $TMP_ROOT"
  rm -rf "$TMP_ROOT" || true
}
trap cleanup EXIT

NPROC="$(getconf _NPROCESSORS_ONLN || echo 2)"

OS_ID="unknown"
if [[ -f /etc/os-release ]]; then
  # shellcheck disable=SC1091
  . /etc/os-release
  OS_ID="${ID:-$OS_ID}"
fi

IS_DEBIAN=false
IS_ARCH=false
if have apt-get; then IS_DEBIAN=true; fi
if have pacman;  then IS_ARCH=true;  fi

$IS_DEBIAN && log "Detected Debian/Ubuntu (ID=${OS_ID})"
$IS_ARCH   && log "Detected Arch Linux"

apt_wait_lock() {
  local tries=0 max_tries=60
  local locks=(/var/lib/dpkg/lock-frontend /var/lib/dpkg/lock /var/lib/apt/lists/lock)
  have fuser || apt-get update >/dev/null 2>&1 || true 
  while :; do
    local busy=0
    for lk in "${locks[@]}"; do
      if fuser "$lk" >/dev/null 2>&1; then busy=1; break; fi
    done
    (( busy==0 )) && break
    (( tries++ ))
    if (( tries > max_tries )); then
      log "APT still locked; attempting gentle recover…"
      pkill -f unattended-upgrade || true
      break
    fi
    printf "."
    sleep 5
  done
  echo
}

install_deps_debian() {
  apt_wait_lock
  log "Updating APT indexes"
  apt-get update -y

  log "Installing core build deps + libraries"
  apt-get install -y \
    build-essential cmake git curl pkg-config \
    libboost-all-dev zlib1g-dev libbz2-dev liblzma-dev libeigen3-dev \
    python3-dev python3-venv zstd

  mkdir -p "$HOME/.local/bin"
}

install_deps_arch() {
  log "Syncing pacman and installing deps"
  sudo pacman -Syu --noconfirm
  sudo pacman -S --noconfirm --needed \
    base-devel cmake git curl pkgconf \
    boost zlib bzip2 xz eigen python zstd
  mkdir -p "$HOME/.local/bin"
}

if $IS_DEBIAN; then install_deps_debian
elif $IS_ARCH; then install_deps_arch
else die "Unsupported distro. Needs apt (Debian/Ubuntu) or pacman (Arch)."
fi

if ! have uv; then
  log "Installing uv (Python package manager)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
else
  log "uv already installed"
fi

export PATH="$HOME/.local/bin:$PATH"
add_line_once 'export PATH="$HOME/.local/bin:$PATH"' "$SHELL_RC"

build_kenlm() {
  log "Building KenLM from source"
  local work="$TMP_ROOT/kenlm"
  git clone --depth 1 https://github.com/kpu/kenlm.git "$work"
  cmake -S "$work" -B "$work/build" -DCMAKE_BUILD_TYPE=Release
  cmake --build "$work/build" -j"$NPROC"
  cmake --install "$work/build"
  cp -f "$work/build/bin/lmplz" "$work/build/bin/build_binary" /usr/local/bin/ 2>/dev/null || true

  if [[ -x /usr/local/bin/lmplz && -x /usr/local/bin/build_binary ]]; then
    KENLM_DIR="/usr/local/bin"
  else
    KENLM_DIR="$work/build/bin"
  fi
  log "Setting KENLM_BIN=$KENLM_DIR"
  export KENLM_BIN="$KENLM_DIR"
  add_line_once "export KENLM_BIN=\"$KENLM_DIR\"" "$SHELL_RC"
}

build_kenlm

build_cmix() {
  log "Building cmix (using g++)"
  local work="$TMP_ROOT/cmix"
  git clone --depth 1 https://github.com/byronknoll/cmix.git "$work"
  if grep -qE '^CC *= *' "$work/makefile"; then
    sed -i.bak 's/^CC *= *.*/CC = g++/' "$work/makefile"
  else
    sed -i.bak 's/^CXX *= *.*/CXX = g++/' "$work/makefile" || true
  fi
  make -C "$work" -j"$NPROC"
  mv "$work/cmix" /usr/local/bin/
  log "cmix installed to /usr/local/bin/cmix"
}

build_cmix

if [[ -f "Makefile" && -n "$(grep -E '(^|\s)dev:|uv sync' Makefile || true)" ]]; then
  log "Syncing project dependencies via Makefile (dev group)"
  make dev
else
  log "Syncing project dependencies via uv directly"
  uv sync --group dev || uv sync
fi

add_line_once 'export KENLM_BIN="${KENLM_BIN:-/usr/local/bin}"' "$SHELL_RC"
add_line_once 'hash -r 2>/dev/null || true' "$SHELL_RC"

if $IS_DEBIAN; then
  log "Cleaning APT cache"
  apt-get clean -y || true
fi

log "Verifying toolchain…"
which uv              || die "uv not on PATH"
which zstd            || die "zstd missing"
which lmplz           || die "KenLM lmplz missing"
which build_binary    || die "KenLM build_binary missing"
which cmix            || die "cmix missing"
log "All binaries present."

log "Done! Open a new shell or run:  source \"$SHELL_RC\""
cat <<'NEXT'

Quick checks:
  lmplz --help          # should print usage
  build_binary -h       # should print usage
  cmix                  # should print usage

Kick off your experiment:
  make run-exp ARGS="+experiment=e1a_wiki103"

Notes:
- KENLM_BIN is set to the directory containing lmplz/build_binary.
- cmix is built with g++ and installed to /usr/local/bin.
- uv synced your dev deps from pyproject.toml / Makefile.
- If you use zsh, we appended exports to ~/.zshrc; otherwise ~/.bashrc.

NEXT


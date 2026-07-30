#!/usr/bin/env bash
# Full setup script for expert-iter + kimina-lean-server on a new Ubuntu server.
# Tested on: Ubuntu 22.04, CUDA 12.x, Python 3.11, A6000/A4000
# Usage: bash scripts/setup_new_server.sh [--skip-mathlib] [--skip-lean]
#
#   --skip-lean     Math-only setup: install the expert-iter venv and stop.
#                   Skips Elan/Lean, kimina-lean-server, and the smoke test.
#                   Use this when the run only needs math-verify for grading.
#   --skip-mathlib  Install Lean + kimina but skip the ~50 GB Mathlib build.
#
# Safe to run from any working directory: the expert-iter checkout is located
# from this script's own path, not from $PWD or $HOME. Override with
# EXPERT_ITER_DIR=... only if you want to set up a *different* checkout.
set -euo pipefail

SKIP_MATHLIB=false
SKIP_LEAN=false
for arg in "$@"; do
  [[ "$arg" == "--skip-mathlib" ]] && SKIP_MATHLIB=true
  [[ "$arg" == "--skip-lean" ]] && SKIP_LEAN=true
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERT_ITER_DIR="${EXPERT_ITER_DIR:-$REPO_ROOT}"
KIMINA_DIR="${KIMINA_DIR:-$HOME/kimina-lean-server}"
LEAN_VERSION="v4.26.0"

log() { echo -e "\n\033[1;32m>>> $*\033[0m"; }
die() { echo -e "\033[1;31mERROR: $*\033[0m" >&2; exit 1; }

# Drop a venv that can't actually run: a dangling interpreter symlink (checkout
# copied between machines, uv-managed python removed, $HOME moved) or the wrong
# Python minor. Bare `[ -d .venv ]` guards miss both and fail later at install.
ensure_clean_venv() {
  local venv_dir="$1"
  [ -d "$venv_dir" ] || return 0
  if ! "$venv_dir/bin/python" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 11) else 1)' 2>/dev/null; then
    log "Removing stale/incompatible venv at $venv_dir"
    rm -rf "$venv_dir"
  fi
}

# ─── 0. System prerequisites (no sudo required) ─────────────────────────────
# Everything here installs into $HOME/.local — no root/apt needed. git and curl
# are the only hard prerequisites; if missing they need a one-time sudo install
# by an admin (nothing else in this script does).
log "Checking system prerequisites"
command -v git  >/dev/null || die "git not found (ask admin: sudo apt install git)"
command -v curl >/dev/null || die "curl not found (ask admin: sudo apt install curl)"

mkdir -p "$HOME/.local/bin"
export PATH="$HOME/.local/bin:$PATH"

# uv (fast venv + pip replacement) — installed FIRST so it can provision Python
if ! command -v uv >/dev/null 2>&1; then
  log "Installing uv (user-local, no sudo)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# Python 3.11 (required for flash-attn wheel) — provisioned via uv, no sudo.
# The symlink makes the later `command -v python3.11` / `uv venv --python
# python3.11` checks resolve to the uv-managed interpreter.
if ! command -v python3.11 >/dev/null 2>&1; then
  log "Installing Python 3.11 via uv (user-local, no sudo)"
  uv python install 3.11
  ln -sf "$(uv python find 3.11)" "$HOME/.local/bin/python3.11"
fi

# jq (kimina setup.sh may use it) — static binary into ~/.local/bin, no sudo
if ! command -v jq >/dev/null 2>&1; then
  log "Installing jq (user-local static binary, no sudo)"
  curl -Lsf https://github.com/jqlang/jq/releases/latest/download/jq-linux-amd64 \
    -o "$HOME/.local/bin/jq" && chmod +x "$HOME/.local/bin/jq"
fi

# ─── 1. expert-iter venv ────────────────────────────────────────────────────
log "Setting up expert-iter venv at $EXPERT_ITER_DIR/.venv"
[ -f "$EXPERT_ITER_DIR/pyproject.toml" ] \
  || die "$EXPERT_ITER_DIR does not look like the expert-iter checkout (no pyproject.toml)"
cd "$EXPERT_ITER_DIR"

ensure_clean_venv "$EXPERT_ITER_DIR/.venv"

# `uv sync` creates .venv itself (honouring requires-python = "==3.11.*") and
# makes it match the lock exactly, removing anything extraneous. With a
# committed uv.lock, --frozen skips re-resolution entirely — important here
# because torch/vllm/flash-attn ABIs are pinned to each other and a fresh
# resolve on a new day can quietly pick different transitive versions.
log "Installing expert-iter dependencies (this may take ~10 min for flash-attn)"
if [ -f uv.lock ]; then
  uv sync --frozen --python python3.11
else
  uv sync --python python3.11
  log "Generated uv.lock — commit it so future servers get this exact tree."
fi

# Math-only mode: everything below is the Lean proof-verification stack
# (Elan toolchain, kimina server, Mathlib, smoke test). Math grading goes
# through the math-verify package, which uv sync above already installed.
if $SKIP_LEAN; then
  log "Setup complete (math-only, --skip-lean)!"
  cat <<EOF

  expert-iter venv : $EXPERT_ITER_DIR/.venv

  Lean/kimina were NOT installed. Math answer checking (math-verify) works
  as-is. If you later need Lean proof verification, re-run without
  --skip-lean; the venv step is idempotent and will be fast the second time.

  The launchers under scripts/ and eval/ activate the expert-iter venv
  themselves, so you do NOT need to source it first. For ad-hoc python:
    source $EXPERT_ITER_DIR/.venv/bin/activate
EOF
  exit 0
fi

# ─── 2. Elan + Lean toolchain ───────────────────────────────────────────────
log "Installing Elan + Lean ${LEAN_VERSION}"
if ! command -v elan >/dev/null 2>&1; then
  curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf \
    | sh -s -- --default-toolchain "${LEAN_VERSION}" -y
fi
source "$HOME/.elan/env"
lean --version

# ─── 3. kimina-lean-server ──────────────────────────────────────────────────
log "Cloning kimina-lean-server"
if [ ! -d "$KIMINA_DIR" ]; then
  git clone https://github.com/project-numina/kimina-lean-server.git "$KIMINA_DIR"
fi
cd "$KIMINA_DIR"

# Write .env if it doesn't exist
if [ ! -f .env ]; then
  log "Writing kimina .env"
  cat > .env <<'EOF'
LEAN_SERVER_HOST=0.0.0.0
LEAN_SERVER_PORT=8000
LEAN_SERVER_LOG_LEVEL=INFO
LEAN_SERVER_ENVIRONMENT=dev
LEAN_SERVER_LEAN_VERSION=v4.26.0
LEAN_SERVER_MAX_REPLS=
LEAN_SERVER_MAX_REPL_USES=-1
LEAN_SERVER_MAX_REPL_MEM=12G
LEAN_SERVER_MAX_WAIT=60
LEAN_SERVER_INIT_REPLS={}
EOF
fi

log "Running kimina setup.sh (builds Lean REPL + optionally Mathlib)"
if $SKIP_MATHLIB; then
  log "  --skip-mathlib set: skipping Mathlib download (~50 GB)"
  # Run setup.sh but stop before mathlib; set env var if supported
  SKIP_MATHLIB_BUILD=true bash setup.sh || true
else
  bash setup.sh
fi

# kimina Python venv
log "Setting up kimina venv"
ensure_clean_venv "$KIMINA_DIR/.venv"
if [ ! -d ".venv" ]; then
  uv venv .venv --python python3.11
fi
# The server runtime deps (fastapi/uvicorn/prisma/...) live in the `server`
# extra; `dev` is a [tool.uv] dev-dependencies group, not a PEP621 extra, so
# `-e .[dev]` silently skips uvicorn and the server won't start.
uv pip install --python .venv/bin/python -e ".[server]"

# The server imports a Prisma client that must be generated first, else
# `python -m server` dies with "Client hasn't been generated yet". The prisma
# CLI shells out to the `prisma-client-py` generator, so .venv/bin must be on
# PATH for generate to find it.
log "Generating Prisma client"
PATH="$KIMINA_DIR/.venv/bin:$PATH" .venv/bin/python -m prisma generate

# ─── 4. Smoke test ──────────────────────────────────────────────────────────
log "Starting kimina server for smoke test (background)"
source "$HOME/.elan/env"
cd "$KIMINA_DIR"
nohup .venv/bin/python -m server > /tmp/kimina_server.log 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

log "Waiting for server to be ready..."
for i in $(seq 1 30); do
  if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
    log "Server is up!"
    break
  fi
  sleep 2
  if [ $i -eq 30 ]; then
    kill $SERVER_PID 2>/dev/null || true
    die "Server did not start in 60s. Check /tmp/kimina_server.log"
  fi
done

log "Sending test verification request"
# Each code entry needs a custom_id (server rejects with 422 otherwise). The
# `import Mathlib` + norm_num body doubles as a check that the Mathlib cache is
# wired up (norm_num is a Mathlib tactic). Empty "messages" == proof closed.
RESPONSE=$(curl -sf -X POST http://localhost:8000/verify \
  -H "Content-Type: application/json" \
  -d '{"codes": [{"custom_id": "smoke", "code": "import Mathlib\ntheorem test : 1 + 1 = 2 := by norm_num"}]}')
echo "Response: $RESPONSE"

kill $SERVER_PID 2>/dev/null || true

# ─── 5. Summary ─────────────────────────────────────────────────────────────
log "Setup complete!"
cat <<EOF

  expert-iter venv : $EXPERT_ITER_DIR/.venv
  kimina venv      : $KIMINA_DIR/.venv
  Lean toolchain   : $HOME/.elan/

  The launchers under scripts/ and eval/ activate the expert-iter venv
  themselves, so you do NOT need to source it first. For ad-hoc python:
    source $EXPERT_ITER_DIR/.venv/bin/activate

  To start kimina server:
    cd $KIMINA_DIR
    source ~/.elan/env
    .venv/bin/python -m server

  To test from expert-iter (custom_id is required):
    curl -X POST http://localhost:8000/verify \\
      -H "Content-Type: application/json" \\
      -d '{"codes": [{"custom_id": "t", "code": "theorem t : 1+1=2 := rfl"}]}'
EOF
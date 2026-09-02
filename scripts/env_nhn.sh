#!/usr/bin/env bash
# NHN GPU-hub container environment.  SOURCE it, don't execute it:
#
#   source scripts/env_nhn.sh
#
# The container filesystem is recycled (the hub stops a container after ~2 days
# and it must be re-registered in the queue), so EVERYTHING that is expensive to
# rebuild has to live on the NAS volume — the "내 워크스페이스" entry the console
# shows as `<workspace>/BASE -> /NHNHOME/WORKSPACE/<workspace>`.  That mount is
# the only path that survives; /root, /home, /tmp and the image layers do not.
#
# Override the mount for another workspace with:  EI_BASE=/NHNHOME/WORKSPACE/xxx
#
# SECRETS ARE NEVER COMMITTED.  This file only *reads* $EI_BASE/secrets.env,
# which lives on the NAS and is created once per workspace:
#
#   printf 'WANDB_API_KEY=%s\n' '<key from https://wandb.ai/authorize>' \
#       > "$EI_BASE/secrets.env" && chmod 600 "$EI_BASE/secrets.env"

export EI_BASE="${EI_BASE:-/NHNHOME/WORKSPACE/26msit001_A}"

if [ ! -d "$EI_BASE" ]; then
  echo "[env] WARNING: $EI_BASE is not mounted — caches would land on the" >&2
  echo "[env]          container's ephemeral disk and vanish on recycle." >&2
fi

# Caches: a re-created container re-uses these instead of re-downloading the
# 8 GB base model, the benchmark datasets, and the pinned vllm/flash-attn wheels.
export HF_HOME="$EI_BASE/hf_cache"
export UV_CACHE_DIR="$EI_BASE/uv_cache"
export VLLM_CACHE_ROOT="$EI_BASE/vllm_cache"
export TRITON_CACHE_DIR="$EI_BASE/triton_cache"

# uv itself and the Python 3.11 it provisions must ALSO live on the NAS.
# setup.sh installs both under $HOME by default, and `.venv/bin/python` is a
# symlink into the interpreter's install dir — so a NAS-resident .venv whose
# interpreter sat in the container's $HOME comes back DANGLING after a recycle.
export UV_INSTALL_DIR="$EI_BASE/bin"
export UV_PYTHON_INSTALL_DIR="$EI_BASE/uv_python"
export PATH="$UV_INSTALL_DIR:$PATH"

mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" \
         "$UV_INSTALL_DIR" "$UV_PYTHON_INSTALL_DIR" 2>/dev/null || true

# WANDB_API_KEY (and anything else secret) from the NAS-only file.  Without it
# wandb_utils silently downgrades an `online` config to offline, and offline
# runs ignore the run id — every container restart then starts a NEW wandb run.
if [ -f "$EI_BASE/secrets.env" ]; then
  set -a; . "$EI_BASE/secrets.env"; set +a
else
  echo "[env] no $EI_BASE/secrets.env — wandb will run offline; sync later with" >&2
  echo "[env]   wandb sync runs/<name>/wandb/offline-run-*" >&2
fi

echo "[env] EI_BASE=$EI_BASE  HF_HOME=$HF_HOME  wandb_key=$([ -n "${WANDB_API_KEY:-}" ] && echo set || echo unset)"

"""LoRA adapter -> full-model resolution for inference stages.

Our own training saves full checkpoints, but externally-trained checkpoints
(e.g. OPSD's) are often PEFT adapter dirs. Rather than teach every vLLM pool
worker about LoRA (rank caps, adapter plumbing), we merge the adapter into its
base model ONCE, cache the result next to the adapter, and hand the merged dir
to the unchanged engine path. Full models and hub ids pass through untouched.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .utils import is_done, mark_done, stable_hash


def resolve_model_path(model_path: str, dtype: str = "bfloat16") -> str:
    """Return a path vLLM can load directly.

    If model_path is a PEFT adapter dir (has adapter_config.json), merge into
    <adapter_dir>/merged/<content-key>/ and return that; otherwise return
    model_path unchanged.
    """
    adapter_dir = Path(model_path)
    if not (adapter_dir / "adapter_config.json").exists():
        return model_path

    cache_key = _adapter_cache_key(adapter_dir, dtype)
    merged = adapter_dir / "merged" / cache_key
    if is_done(merged / "config.json", config_hash=cache_key):
        print(f"[lora] using cached merge {merged}")
        return str(merged)

    try:
        from peft import PeftModel
    except ImportError as e:
        raise RuntimeError(
            f"{model_path} is a LoRA adapter dir but peft is not installed; "
            "run `uv sync` (peft is in pyproject) or pass a merged/full checkpoint."
        ) from e

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = json.loads((adapter_dir / "adapter_config.json").read_text())["base_model_name_or_path"]
    print(f"[lora] merging adapter {adapter_dir} into base {base} -> {merged}")
    torch_dtype = "auto" if dtype == "auto" else getattr(torch, dtype)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch_dtype)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model = model.merge_and_unload()
    model.save_pretrained(merged)
    # vLLM loads the tokenizer from the model dir; adapters ship without one.
    AutoTokenizer.from_pretrained(base).save_pretrained(merged)
    mark_done(merged / "config.json", count=1, config_hash=cache_key)
    return str(merged)


def fmt_alpha(alpha: float) -> str:
    """Directory-name form of a scaling factor: 0.05 -> '0.05', 1.0 -> '1'."""
    return f"{alpha:g}"


def make_alpha_variant(adapter_dir: str | Path, alpha: float) -> Path:
    """Materialize q_alpha = pi_{theta + alpha*phi} as an adapter dir.

    LoRA's effective scale is lora_alpha / r, so an adapter_config.json copy
    with lora_alpha multiplied by alpha IS the alpha-scaled adapter — the
    weight files are shared via hardlink (fallback: copy). alpha == 1.0
    returns the base dir untouched. Variants live under
    <adapter_dir>/alpha/<alpha>/ so the content-addressed parent path keeps
    engine pool cache keys honest. Idempotent."""
    import shutil

    adapter_dir = Path(adapter_dir)
    if alpha == 1.0:
        return adapter_dir
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1]: {alpha}")
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"not an adapter dir: {adapter_dir}")
    out = adapter_dir / "alpha" / fmt_alpha(alpha)
    cfg = json.loads(config_path.read_text())
    scaled = cfg["lora_alpha"] * alpha
    out_cfg = out / "adapter_config.json"
    if out_cfg.exists() and json.loads(out_cfg.read_text()).get("lora_alpha") == scaled:
        return out
    out.mkdir(parents=True, exist_ok=True)
    for p in adapter_dir.iterdir():
        if not p.is_file() or p.name == "adapter_config.json":
            continue
        dst = out / p.name
        if dst.exists():
            continue
        try:
            dst.hardlink_to(p)
        except OSError:
            shutil.copy2(p, dst)
    cfg["lora_alpha"] = scaled
    out_cfg.write_text(json.dumps(cfg, indent=2))
    return out


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _adapter_cache_key(adapter_dir: Path, dtype: str) -> str:
    """Fingerprint config and actual adapter weights, not merely their path."""
    config_path = adapter_dir / "adapter_config.json"
    weights = sorted(
        p for p in adapter_dir.glob("adapter_model*") if p.is_file()
    )
    if not weights:
        raise RuntimeError(f"LoRA adapter has no adapter_model weights: {adapter_dir}")
    manifest = [
        (p.name, p.stat().st_size, _file_sha256(p))
        for p in weights
    ]
    return stable_hash(
        "lora_merge_v2", dtype, _file_sha256(config_path), manifest,
    )

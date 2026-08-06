"""LoRA adapter -> full-model resolution for inference stages.

Our own training saves full checkpoints, but externally-trained checkpoints
(e.g. OPSD's) are often PEFT adapter dirs. Rather than teach every vLLM pool
worker about LoRA (rank caps, adapter plumbing), we merge the adapter into its
base model ONCE, cache the result next to the adapter, and hand the merged dir
to the unchanged engine path. Full models and hub ids pass through untouched.
"""

from __future__ import annotations

from pathlib import Path

from .utils import is_done, mark_done, stable_hash


def resolve_model_path(model_path: str, dtype: str = "bfloat16") -> str:
    """Return a path vLLM can load directly.

    If model_path is a PEFT adapter dir (has adapter_config.json), merge into
    <adapter_dir>/merged/ (cached via .done marker keyed on adapter+base) and
    return that; otherwise return model_path unchanged.
    """
    adapter_dir = Path(model_path)
    if not (adapter_dir / "adapter_config.json").exists():
        return model_path

    merged = adapter_dir / "merged"
    cache_key = stable_hash("lora_merge", str(adapter_dir.resolve()), dtype)
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
    import json

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = json.loads((adapter_dir / "adapter_config.json").read_text())["base_model_name_or_path"]
    print(f"[lora] merging adapter {adapter_dir} into base {base} -> {merged}")
    model = AutoModelForCausalLM.from_pretrained(base, dtype=getattr(torch, dtype))
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model = model.merge_and_unload()
    model.save_pretrained(merged)
    # vLLM loads the tokenizer from the model dir; adapters ship without one.
    AutoTokenizer.from_pretrained(base).save_pretrained(merged)
    mark_done(merged / "config.json", count=1, config_hash=cache_key)
    return str(merged)

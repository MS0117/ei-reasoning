"""M0: empirically verify the installed API surface before trusting any recipe.

The pinned deps (transformers 5.7, trl 1.3, vllm 0.20, torch 2.11) are newer
than public training recipes, so trainer/collator code is written against what
THIS script prints, not against documentation memory. Run:

    .venv/bin/python scripts/check_env.py [--gpu 2] [--skip-gpu] [--model Qwen/Qwen3-0.6B]

Sections print independently; a failure in one section doesn't stop the rest.
Findings worth keeping go to docs/api_notes.md.
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys
import traceback


def section(title):
    print(f"\n{'=' * 70}\n## {title}\n{'=' * 70}")


def guarded(fn):
    try:
        fn()
    except Exception:
        print(f"!! {fn.__name__} FAILED:")
        traceback.print_exc()


def versions():
    import accelerate
    import datasets
    import torch
    import transformers
    import trl
    import vllm

    print("python      ", sys.version.split()[0])
    print("torch       ", torch.__version__, "| cuda", torch.version.cuda,
          "| devices", torch.cuda.device_count())
    print("transformers", transformers.__version__)
    print("trl         ", trl.__version__)
    print("accelerate  ", accelerate.__version__)
    print("vllm        ", vllm.__version__)
    print("datasets    ", datasets.__version__)
    try:
        import deepspeed

        print("deepspeed   ", deepspeed.__version__)
    except Exception as e:
        print("deepspeed    IMPORT FAILED:", e)
    try:
        import flash_attn

        print("flash_attn  ", flash_attn.__version__)
    except Exception as e:
        print("flash_attn   IMPORT FAILED:", e)


def trl_sft_surface():
    from trl import SFTConfig, SFTTrainer

    print("SFTTrainer.__init__:", inspect.signature(SFTTrainer.__init__))
    print("\nSFTTrainer MRO:", [c.__name__ for c in SFTTrainer.__mro__[:4]])
    fields = sorted(
        f for f in SFTConfig.__dataclass_fields__
        if any(k in f for k in (
            "packing", "padding", "completion", "loss", "dataset", "assistant",
            "max_length", "seq", "group_by", "model_init", "chat",
        ))
    )
    print("\nSFTConfig relevant fields:")
    for f in fields:
        fld = SFTConfig.__dataclass_fields__[f]
        default = getattr(fld, "default", None)
        print(f"  {f} = {default!r}")
    has_compute_loss = "compute_loss" in SFTTrainer.__dict__
    print("\nSFTTrainer defines its own compute_loss:", has_compute_loss)
    if has_compute_loss:
        print("  signature:", inspect.signature(SFTTrainer.compute_loss))


def trl_dpo_surface():
    from trl import DPOConfig, DPOTrainer

    print("DPOTrainer.__init__:", inspect.signature(DPOTrainer.__init__))
    fields = sorted(
        f for f in DPOConfig.__dataclass_fields__
        if any(k in f for k in ("beta", "loss_type", "max_length", "max_prompt", "label_pad", "precompute"))
    )
    print("DPOConfig relevant fields:")
    for f in fields:
        print(f"  {f} = {getattr(DPOConfig.__dataclass_fields__[f], 'default', None)!r}")


def transformers_loss_contract():
    from transformers import Trainer

    sig = inspect.signature(Trainer.compute_loss)
    print("Trainer.compute_loss:", sig)
    print("has num_items_in_batch param:", "num_items_in_batch" in sig.parameters)
    src = inspect.getsource(Trainer.get_batch_samples)
    print("\nTrainer.get_batch_samples source (num_items accounting):")
    print(src)


def vllm_surface():
    import vllm
    from vllm import SamplingParams

    sp = inspect.signature(SamplingParams.__init__)
    wanted = ["n", "temperature", "top_p", "max_tokens", "seed", "logprobs",
              "prompt_logprobs", "stop", "include_stop_str_in_output", "detokenize"]
    print("SamplingParams params of interest:")
    for w in wanted:
        print(f"  {w}: {'YES' if w in sp.parameters else 'MISSING'}")
    try:
        from vllm.inputs import TokensPrompt

        print("vllm.inputs.TokensPrompt: available")
    except ImportError:
        print("vllm.inputs.TokensPrompt: MISSING — check alternative prompt-ids API")
    llm_sig = inspect.signature(vllm.LLM.__init__)
    for w in ["enable_prefix_caching", "gpu_memory_utilization", "max_model_len",
              "tensor_parallel_size", "seed", "enforce_eager", "dtype"]:
        print(f"  LLM.{w}: {'YES' if w in llm_sig.parameters else 'MISSING'}")


def qwen_template(model_id: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    msgs = [{"role": "user", "content": "What is 2+2?"}]
    rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    print(f"chat template of {model_id} (add_generation_prompt=True):")
    print(repr(rendered))
    print("\ncontains '<think>':", "<think>" in rendered)
    ids_text = tok(rendered, add_special_tokens=False)["input_ids"]
    ids_direct = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    print("apply_chat_template(tokenize=True) returns:", type(ids_direct).__name__,
          "(transformers 5.x: BatchEncoding, NOT a flat id list — templates.py "
          "therefore encodes the rendered text instead)")
    if hasattr(ids_direct, "get"):
        ids_direct = ids_direct.get("input_ids", ids_direct)
        if ids_direct and isinstance(ids_direct[0], list):
            ids_direct = ids_direct[0]
    print("tokenize=True matches encode(text):", ids_text == list(ids_direct))
    print("eos_token:", repr(tok.eos_token), tok.eos_token_id)
    # Splice-safety probe: does encode(decode(slice)) round-trip?
    resp_ids = tok("The answer is computed as follows: 1+1=2, so", add_special_tokens=False)["input_ids"]
    cut = resp_ids[: len(resp_ids) // 2]
    reenc = tok(tok.decode(cut), add_special_tokens=False)["input_ids"]
    print("anchor decode->re-encode roundtrip equal:", reenc == cut,
          "(False is EXPECTED sometimes — this is why we splice ids, not text)")


def gpu_generation(model_id: str, gpu: int):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpu))
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(model=model_id, dtype="bfloat16", gpu_memory_utilization=0.4,
              max_model_len=2048, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    msgs = [{"role": "user", "content": "What is 2+2? Answer with just the number."}]
    # encode rendered text — apply_chat_template(tokenize=True) returns a
    # BatchEncoding in transformers 5.x and list(...) would yield dict keys
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    prompt_ids = tok(text, add_special_tokens=False)["input_ids"]

    # 1) generation from token ids (the continuation-from-prefix path)
    out = llm.generate(
        [TokensPrompt(prompt_token_ids=list(prompt_ids))],
        SamplingParams(max_tokens=10, temperature=0.0, seed=1234),
    )[0].outputs[0]
    print("generated text:", repr(out.text))
    print("output token_ids len:", len(out.token_ids), "finish_reason:", out.finish_reason)

    # 2) prompt_logprobs scoring (the logprob-gate path)
    seq = list(prompt_ids) + list(out.token_ids)
    scored = llm.generate(
        [TokensPrompt(prompt_token_ids=seq)],
        SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0),
    )[0]
    plp = scored.prompt_logprobs
    print("prompt_logprobs: len", len(plp) if plp else None,
          "| pos0 is None:", plp[0] is None if plp else "n/a")
    if plp and len(plp) > 1:
        tail = [next(iter(d.values())).logprob for d in plp[-5:] if d]
        print("last-5 token logprobs:", [round(x, 3) for x in tail])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=None, help="GPU id for the live test")
    ap.add_argument("--skip-gpu", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--template-model", default="Qwen/Qwen3-4B-Instruct-2507")
    args = ap.parse_args()

    section("versions"); guarded(versions)
    section("trl SFTTrainer/SFTConfig"); guarded(trl_sft_surface)
    section("trl DPOTrainer/DPOConfig"); guarded(trl_dpo_surface)
    section("transformers loss contract"); guarded(transformers_loss_contract)
    section("vllm surface"); guarded(vllm_surface)
    section("chat template"); guarded(lambda: qwen_template(args.template_model))
    if not args.skip_gpu:
        gpu = args.gpu if args.gpu is not None else 0
        section(f"live GPU generation ({args.model} on GPU {gpu})")
        guarded(lambda: gpu_generation(args.model, gpu))


if __name__ == "__main__":
    main()

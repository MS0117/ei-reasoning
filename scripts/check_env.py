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


def trl_grpo_surface():
    """CPU surface for improve.rl: GRPO/RLOO trainers, colocate-vllm knobs, and
    the trl<->vllm version-band warning (trl officially supports <= 0.18; the
    live behavior on the pinned vllm is checked by gpu_trl_grpo_probe)."""
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from trl import GRPOConfig, GRPOTrainer, RLOOConfig, RLOOTrainer
    for w in caught:
        if "vLLM" in str(w.message):
            print("trl<->vllm band warning:", w.message)
    print("GRPOTrainer.__init__:", inspect.signature(GRPOTrainer.__init__))
    print("RLOOTrainer.__init__:", inspect.signature(RLOOTrainer.__init__))
    wanted = ["use_vllm", "vllm_mode", "vllm_enable_sleep_mode",
              "vllm_gpu_memory_utilization", "num_generations", "epsilon", "beta",
              "num_iterations", "loss_type", "max_completion_length",
              "mask_truncated_completions", "scale_rewards"]
    for cls in (GRPOConfig, RLOOConfig):
        fields = cls.__dataclass_fields__
        print(f"\n{cls.__name__} fields of interest:")
        for w in wanted:
            default = getattr(fields[w], "default", "MISSING") if w in fields else "MISSING"
            print(f"  {w}: {default!r}" if w in fields else f"  {w}: MISSING")


def _is_distributed_launch() -> bool:
    """True when running under `accelerate launch` / torchrun (each rank owns
    one GPU and device assignment is the launcher's job)."""
    return "LOCAL_RANK" in os.environ or int(os.environ.get("WORLD_SIZE", "1")) > 1


def gpu_trl_grpo_probe(model_id: str, gpu: int):
    """Live probe for the improve.rl trl backend on THIS stack (vllm 0.20 is
    outside trl's supported band): warm-start a real lora_fit adapter, run
    GRPOTrainer 2 steps with colocate vLLM + sleep mode (exercises
    merge_adapter -> load_weights -> llm.sleep/wake_up), then re-serve the
    trained adapter through a plain vLLM LoRARequest. Green => trust
    improve.rl.backend=trl; red => the pool fallback must land first.

    Single-process run = the shipped path (lora_rl.py pins one GPU). The SAME
    probe under `accelerate launch --num_processes N scripts/check_env.py
    --only rl` validates the multi-GPU (DDP + per-rank colocate vLLM) path
    that a future multi-GPU lora_rl would use — trl supports multi-GPU only
    via DDP, never via the Trainer's implicit DataParallel."""
    import json
    import subprocess
    import tempfile
    from pathlib import Path

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if _is_distributed_launch():
        print(f"[probe] distributed launch: rank {rank}/{world} "
              f"(device assignment left to accelerate; CVD={os.environ.get('CUDA_VISIBLE_DEVICES')})")
    else:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpu))
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from expert_iter.utils import write_jsonl

    scratch = Path(tempfile.mkdtemp(prefix=f"rl_probe_r{rank}_"))
    tok = AutoTokenizer.from_pretrained(model_id)
    prompt_ids = tok("Question: what is 2+2?\nAnswer:", add_special_tokens=False)["input_ids"]
    sol_ids = tok(" The answer is 4.", add_special_tokens=False)["input_ids"]
    pairs_path = scratch / "pairs.jsonl"
    write_jsonl(pairs_path, [{
        "qid": "probe",
        "input_ids": list(prompt_ids) + list(sol_ids) + [tok.eos_token_id],
        "prompt_len": len(prompt_ids),
    }])
    adapter = scratch / "adapter"
    fit_params = {"r": 16, "lora_alpha": 32, "lr": 1e-4, "steps": 1, "seed": 0,
                  "gradient_checkpointing": False, "attn_implementation": "eager"}
    print(f"[1] lora_fit subprocess -> {adapter}")
    fit_env = {**os.environ}
    if not _is_distributed_launch():
        fit_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    else:
        # Each rank runs its own single-GPU child. EVERY distributed marker must
        # go: a leftover TORCHELASTIC_*/MASTER_* makes the child's vLLM attempt a
        # torch.distributed rendezvous and hang for its 600 s timeout.
        fit_env["CUDA_VISIBLE_DEVICES"] = os.environ.get("LOCAL_RANK", "0")
        for k in list(fit_env):
            if k.startswith(("TORCHELASTIC_", "ACCELERATE_")) or k in (
                "LOCAL_RANK", "RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK",
                "ROLE_RANK", "ROLE_NAME", "ROLE_WORLD_SIZE", "GROUP_WORLD_SIZE",
                "MASTER_ADDR", "MASTER_PORT", "TORCH_NCCL_ASYNC_ERROR_HANDLING",
            ):
                fit_env.pop(k, None)
    subprocess.run(
        [sys.executable, "-m", "expert_iter.lora_fit", "--model", model_id,
         "--pairs", str(pairs_path), "--out", str(adapter),
         "--params-json", json.dumps(fit_params), "--cache-key", "rlprobe"],
        env=fit_env, check=True,
    )

    print(f"[2] GRPOTrainer colocate 2 steps, world_size={world} "
          "(merge_adapter/load_weights/sleep path)...")
    base = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
    policy = PeftModel.from_pretrained(base, str(adapter), is_trainable=True)
    args_ = GRPOConfig(
        output_dir=str(scratch / "trainer"), max_steps=2, learning_rate=1e-5,
        per_device_train_batch_size=4, num_generations=4,
        use_vllm=True, vllm_mode="colocate", vllm_enable_sleep_mode=True,
        vllm_gpu_memory_utilization=0.3, max_completion_length=32,
        temperature=1.0, bf16=True, save_strategy="no", report_to=[],
        logging_steps=1, seed=0,
    )
    rows = [{"prompt": "Question: what is 2+2?\nAnswer:"} for _ in range(4)]
    trainer = GRPOTrainer(
        model=policy,
        reward_funcs=lambda prompts, completions, **kw: [
            1.0 if "4" in c else 0.0 for c in completions],
        args=args_, train_dataset=Dataset.from_list(rows), processing_class=tok,
    )
    trainer.train()
    print("   train() completed | peak CUDA mem:",
          f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    out = scratch / "adapter_rl"
    trainer.model.save_pretrained(str(out))
    print("[3] trained adapter saved:", (out / "adapter_config.json").exists())

    del trainer, policy, base
    torch.cuda.empty_cache()
    if rank != 0:
        print("[4] skipped on non-zero rank (serving check runs once)")
        return
    # Serve the RL adapter from a FRESH process, mirroring production: lora_rl
    # exits after saving and the candidate pool boots its own engine.py worker.
    # (Building a second vLLM engine in this process — after trl's colocate
    # engine and its sleep/wake cycle — dies with "CUDA driver error: invalid
    # argument"; that is an in-process teardown artifact, not a serving issue.)
    serve_src = (
        "from vllm import LLM, SamplingParams\n"
        "from vllm.inputs import TokensPrompt\n"
        "from vllm.lora.request import LoRARequest\n"
        f"llm = LLM(model={model_id!r}, dtype='bfloat16', gpu_memory_utilization=0.45,\n"
        "          max_model_len=2048, enable_lora=True, max_lora_rank=16)\n"
        f"o = llm.generate([TokensPrompt(prompt_token_ids={list(prompt_ids)!r})],\n"
        "                 SamplingParams(max_tokens=8, temperature=0.0, seed=7),\n"
        f"                 lora_request=LoRARequest('rl', 1, {str(out)!r}))[0].outputs[0]\n"
        "print('[4] RL adapter serves via LoRARequest:', repr(o.text))\n"
    )
    subprocess.run([sys.executable, "-c", serve_src], env=fit_env, check=True)


def peft_surface():
    import peft
    from peft import LoraConfig

    print("peft        ", peft.__version__)
    sig = inspect.signature(LoraConfig.__init__)
    for w in ["r", "lora_alpha", "lora_dropout", "target_modules", "bias", "task_type"]:
        print(f"  LoraConfig.{w}: {'YES' if w in sig.parameters else 'MISSING'}")
    import vllm

    llm_sig = inspect.signature(vllm.LLM.__init__)
    for w in ["enable_lora", "max_loras", "max_lora_rank", "max_logprobs"]:
        print(f"  LLM.{w}: {'YES' if w in llm_sig.parameters else 'MISSING'} "
              "(MISSING may be a msgspec introspection artifact — trust the live probe)")
    try:
        from vllm.lora.request import LoRARequest  # noqa: F401

        print("vllm.lora.request.LoRARequest: available")
    except ImportError:
        print("vllm.lora.request.LoRARequest: MISSING")


def gpu_lora_probes(model_id: str, gpu: int):
    """Live probes for everything the lora_sft operator relies on:
      1. peft fit + save via the REAL `python -m expert_iter.lora_fit` subprocess;
      2. GPU memory: vLLM at 0.85 utilization immediately after that subprocess
         exits (the improve-stage sequencing);
      3. enable_lora + per-request LoRARequest in generate AND score modes;
      4. the alpha-scaled adapter-config trick (hardlinked weights) and its
         effect direction/magnitude on scored logprobs;
      5. max_logprobs=100 + prompt_logprobs=100 maps, realized token first."""
    import json
    import subprocess
    import tempfile
    from pathlib import Path

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpu))
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from transformers import AutoTokenizer

    from expert_iter.lora import make_alpha_variant
    from expert_iter.utils import write_jsonl

    scratch = Path(tempfile.mkdtemp(prefix="lora_probe_"))
    tok = AutoTokenizer.from_pretrained(model_id)
    prompt_ids = tok("Question: what is 2+2?\nAnswer:", add_special_tokens=False)["input_ids"]
    sol_ids = tok(" The answer is 4.", add_special_tokens=False)["input_ids"]
    pairs_path = scratch / "pairs.jsonl"
    write_jsonl(pairs_path, [{
        "qid": "probe",
        "input_ids": list(prompt_ids) + list(sol_ids) + [tok.eos_token_id],
        "prompt_len": len(prompt_ids),
    }])
    adapter = scratch / "adapter"
    params = {"r": 16, "lora_alpha": 32, "lr": 1e-4, "steps": 2, "seed": 0,
              "gradient_checkpointing": False, "attn_implementation": "eager"}
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    print(f"[1] lora_fit subprocess -> {adapter}")
    subprocess.run(
        [sys.executable, "-m", "expert_iter.lora_fit", "--model", model_id,
         "--pairs", str(pairs_path), "--out", str(adapter),
         "--params-json", json.dumps(params), "--cache-key", "probe"],
        env=env, check=True,
    )
    print("   adapter_config.json:", (adapter / "adapter_config.json").exists(),
          "| adapter_model.safetensors:", (adapter / "adapter_model.safetensors").exists())

    half = make_alpha_variant(adapter, 0.5)
    weights = half / "adapter_model.safetensors"
    print(f"[4] alpha=0.5 variant: {half} | hardlinked:",
          weights.exists() and weights.stat().st_nlink > 1)

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest

    print("[2] constructing vLLM at 0.85 utilization right after the fit subprocess...")
    llm = LLM(model=model_id, dtype="bfloat16", gpu_memory_utilization=0.85,
              max_model_len=2048, enable_lora=True, max_loras=4, max_lora_rank=16,
              max_logprobs=100)
    print("   engine up (memory sequencing OK)")
    lr_full = LoRARequest("full", 1, str(adapter))
    lr_half = LoRARequest("half", 2, str(half))
    seq = list(prompt_ids) + list(sol_ids)

    def score_sum(lora_request=None, k=0):
        out = llm.generate(
            [TokensPrompt(prompt_token_ids=seq)],
            SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=k),
            lora_request=lora_request,
        )[0]
        plp = out.prompt_logprobs or []
        lps = [next(iter(d.values())).logprob for d in plp[len(prompt_ids):] if d]
        return sum(lps), plp

    s_base, _ = score_sum()
    s_full, _ = score_sum(lr_full)
    s_half, _ = score_sum(lr_half)
    print(f"[3] score-mode sum logprob of the trained target — base {s_base:.3f} | "
          f"alpha=1 {s_full:.3f} | alpha=0.5 {s_half:.3f}")
    print("   adapter applied at prefill (full != base):", abs(s_full - s_base) > 1e-3)
    denom = (s_full - s_base) or float("nan")
    print(f"   delta ratio (half-base)/(full-base): {(s_half - s_base) / denom:.3f} "
          "(indicative — softmax is nonlinear in the logit-space alpha scaling)")

    _, plp = score_sum(lr_full, k=100)
    pos = len(prompt_ids)
    d = plp[pos] if plp and len(plp) > pos else None
    if d:
        first_tok = next(iter(d))
        print(f"[5] prompt_logprobs=100: entries at pos {pos} = {len(d)} "
              f"| realized token first: {int(first_tok) == seq[pos]}")

    gen = llm.generate(
        [TokensPrompt(prompt_token_ids=list(prompt_ids))],
        SamplingParams(max_tokens=8, temperature=0.0, seed=7),
        lora_request=lr_full,
    )[0].outputs[0]
    print("[3] generate under adapter:", repr(gen.text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=None, help="GPU id for the live test")
    ap.add_argument("--skip-gpu", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--template-model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--only", default=None, metavar="SECTION",
                    help="run only one GPU probe: gen | lora | rl. `rl` is the "
                         "one that also runs under `accelerate launch "
                         "--num_processes N` to validate multi-GPU (DDP) RL.")
    args = ap.parse_args()

    # Pin the probes to ONE device, overriding an inherited multi-GPU
    # CUDA_VISIBLE_DEVICES (slurm sets e.g. "0,1"). The probes model the
    # production subprocesses (lora_fit / lora_rl), which are always launched
    # on a single GPU; leaving 2 visible would make transformers' Trainer wrap
    # the model in DataParallel — a path trl does not support and we never
    # ship. Under `accelerate launch` the launcher owns device assignment, so
    # the pin is skipped. Must happen before ANY torch import (all local).
    if args.gpu is not None and not _is_distributed_launch():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f">>> probes pinned to CUDA_VISIBLE_DEVICES={args.gpu}")

    gpu = args.gpu if args.gpu is not None else 0
    if args.only:
        probes = {
            "gen": lambda: gpu_generation(args.model, gpu),
            "lora": lambda: gpu_lora_probes(args.model, gpu),
            "rl": lambda: gpu_trl_grpo_probe(args.model, gpu),
        }
        if args.only not in probes:
            ap.error(f"--only must be one of {sorted(probes)}")
        section(f"live GPU probe '{args.only}' ({args.model}, "
                f"world_size={os.environ.get('WORLD_SIZE', '1')})")
        guarded(probes[args.only])
        return

    section("versions"); guarded(versions)
    section("trl SFTTrainer/SFTConfig"); guarded(trl_sft_surface)
    section("trl DPOTrainer/DPOConfig"); guarded(trl_dpo_surface)
    section("transformers loss contract"); guarded(transformers_loss_contract)
    section("vllm surface"); guarded(vllm_surface)
    section("peft / vllm LoRA surface"); guarded(peft_surface)
    section("trl GRPO/RLOO surface"); guarded(trl_grpo_surface)
    section("chat template"); guarded(lambda: qwen_template(args.template_model))
    if not args.skip_gpu:
        section(f"live GPU generation ({args.model} on GPU {gpu})")
        guarded(lambda: gpu_generation(args.model, gpu))
        section(f"live GPU LoRA probes ({args.model} on GPU {gpu})")
        guarded(lambda: gpu_lora_probes(args.model, gpu))
        section(f"live GPU trl GRPO probe ({args.model} on GPU {gpu})")
        guarded(lambda: gpu_trl_grpo_probe(args.model, gpu))


if __name__ == "__main__":
    main()

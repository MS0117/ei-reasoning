"""Transient LoRA SFT — GPU subprocess for the lora_sft improvement operator.

    python -m expert_iter.lora_fit --model <policy dir|hub id> \
        --pairs <pairs.jsonl> --out <adapter_dir> \
        --params-json '{"r":16,"lora_alpha":32,...,"seed":123}' \
        --cache-key <fit_key>

Runs as its own process (repo pattern: GPU phases are subprocesses) so the
parent improve stage never initializes CUDA and the GPU is fully released
before any vLLM pool starts. Plain torch AdamW loop — no Trainer — because 2-4
full-batch gradient steps do not justify the version-volatile surface of the
pinned bleeding-edge trainer stack.

Multi-GPU: launch under `accelerate launch --num_processes N` (the operator
does this when improve.lora_sft.fit.num_processes > 1). Each rank fits its
slice of the pairs under DDP; the local loss carries a world_size factor to
undo DDP's gradient averaging, so the update matches the single-process
full-batch one (mathematically — not bitwise). Rank 0 saves.

Pairs are PRE-TOKENIZED by the operator (templates.gold_pair_ids); this module
never tokenizes text:
    {"qid": str, "input_ids": [int], "prompt_len": int}
Loss applies to positions >= prompt_len only. Every gradient step covers the
FULL pair batch (grad accumulation over micro-batches), so training is
deterministic given the seed — no shuffling order to vary.

Outputs in --out: adapter_config.json + adapter_model.safetensors
(peft save_pretrained), fit_meta.json (loss curve, sizes, seed), and a .done
marker keyed by --cache-key so identical refits are skipped on resume.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .utils import is_done, mark_done, read_jsonl, seed_everything, write_json

DEFAULTS = {
    "r": 16,
    "lora_alpha": 32,
    "lr": 1.0e-4,
    "steps": 3,
    "dropout": 0.0,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "micro_batch_size": 1,
    "max_grad_norm": 1.0,
    "bf16": True,
    "gradient_checkpointing": True,
    "attn_implementation": None,   # None -> flash_attention_2 on CUDA, eager on CPU
    "seed": 0,
    # objective=dpo (staged stage2_objective): pairs carry chosen_ids /
    # rejected_ids (same prompt, prompt_len shared) instead of input_ids. Loss is
    # -logsigmoid(beta * ((pol_c - ref_c) - (pol_r - ref_r))) averaged over the
    # GLOBAL pair count, plus sft_weight * NLL(chosen) per chosen response token.
    # Reference logprobs are precomputed once before training from the fit's
    # starting weights ("init") or with the adapter disabled ("base").
    "objective": "sft",
    "beta": 0.1,
    "sft_weight": 0.0,
    "reference": "init",
}


def _pack(seqs: list[tuple[list[int], int]], device):
    """Right-pad (ids, prompt_len) sequences into ids/mask/labels tensors;
    labels are -100 on the prompt and padding so only response tokens score."""
    import torch

    width = max(len(ids) for ids, _ in seqs)
    ids_t = torch.zeros(len(seqs), width, dtype=torch.long)
    mask = torch.zeros(len(seqs), width, dtype=torch.long)
    labels = torch.full((len(seqs), width), -100, dtype=torch.long)
    for i, (ids, plen) in enumerate(seqs):
        seq = torch.tensor(ids, dtype=torch.long)
        ids_t[i, :len(seq)] = seq
        mask[i, :len(seq)] = 1
        labels[i, plen:len(seq)] = seq[plen:]
    return ids_t.to(device), mask.to(device), labels.to(device)


def _seq_nll(logits, labels):
    """Per-sequence summed NLL over label positions (shifted), and the per-
    sequence label counts. Shared by the SFT loss, the DPO policy pass and the
    DPO reference pass so all three score tokens identically."""
    import torch
    import torch.nn.functional as F

    tgt = labels[:, 1:]
    # one sequence at a time: the fp32 copy of a 16k x 152k logit slab is ~10 GB,
    # and a DPO micro-batch holds two sequences — never materialize both at once
    sums = [F.cross_entropy(logits[i, :-1].float(), tgt[i], ignore_index=-100,
                            reduction="sum") for i in range(logits.shape[0])]
    return torch.stack(sums), (tgt != -100).sum(-1)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="transient LoRA SFT (see module docstring)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--params-json", required=True)
    ap.add_argument("--cache-key", required=True)
    ap.add_argument("--wandb-json", default=None,
                    help="settings for this fit's own wandb run (see "
                         "lora_sft.wandb_params); NOT part of the cache key")
    ap.add_argument("--init-adapter", default=None,
                    help="warm-start from this adapter dir (adaptive tau_E rounds)")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    if is_done(out_dir / "adapter_config.json", config_hash=args.cache_key):
        print(f"[lora_fit] {out_dir} already done, skipping")
        return

    params = {**DEFAULTS, **json.loads(args.params_json)}
    pairs = sorted(read_jsonl(args.pairs), key=lambda p: p["qid"])
    if not pairs:
        raise SystemExit(f"[lora_fit] {args.pairs} is empty — nothing to fit")
    objective = str(params["objective"])
    if objective not in ("sft", "dpo"):
        raise SystemExit(f"[lora_fit] unknown objective {objective!r} (sft | dpo)")
    need = ("chosen_ids", "rejected_ids") if objective == "dpo" else ("input_ids",)
    bad = [i for i, p in enumerate(pairs) if any(k not in p for k in need)]
    if bad:
        raise SystemExit(
            f"[lora_fit] objective={objective} needs pair keys {need}; "
            f"{len(bad)} pair(s) lack them (first at index {bad[0]})"
        )

    seed_everything(int(params["seed"]))

    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    # Data-parallel ranks (accelerate launch). Each rank fits on ITS slice of the
    # pairs; DDP all-reduces (averages) gradients, so the local loss is scaled by
    # world_size to keep the update identical to the single-process full-batch
    # one (same compensation class as api_notes finding 7).
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = torch.cuda.is_available()
    if world > 1:
        torch.distributed.init_process_group(backend="nccl" if use_cuda else "gloo")
        if use_cuda:
            torch.cuda.set_device(local_rank)
    device = (f"cuda:{local_rank}" if use_cuda else "cpu")
    attn = params["attn_implementation"] or ("flash_attention_2" if use_cuda else "eager")
    dtype = torch.bfloat16 if params["bf16"] else torch.float32
    t0 = time.time()
    print(f"[lora_fit] loading {args.model} ({dtype}, {attn}, {device}, "
          f"rank {rank}/{world})")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, attn_implementation=attn,
    ).to(device)
    if args.init_adapter:
        # Warm start (adaptive tau_E rounds): peft 0.20's only warm-start path
        # is PeftModel.from_pretrained; the LoRA shape then comes from the saved
        # adapter_config.json, so a params mismatch must fail loudly.
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
        loaded_r = model.peft_config["default"].r
        if loaded_r != int(params["r"]):
            raise SystemExit(
                f"[lora_fit] --init-adapter r={loaded_r} != params r={params['r']}"
            )
    else:
        model = get_peft_model(model, LoraConfig(
            r=int(params["r"]),
            lora_alpha=int(params["lora_alpha"]),
            lora_dropout=float(params["dropout"]),
            target_modules=list(params["target_modules"]),
            bias="none",
            task_type="CAUSAL_LM",
        ))
    if params["gradient_checkpointing"]:
        model.enable_input_require_grads()
        # use_reentrant=False is required for gradient checkpointing under DDP.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.train()
    peft_model = model
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank] if use_cuda else None,
            # REQUIRED here: gradient checkpointing re-runs the forward during
            # backward, so autograd hooks fire twice per parameter. torch's DDP
            # docs list "activation checkpointing multiple times" as a case
            # static_graph exists for; without it the reducer enqueues
            # reductions the other rank never matches and NCCL deadlocks
            # (observed: rank0 enqueued 1132 works, rank1 1126, 600 s watchdog
            # -> SIGABRT). The graph IS static: same full batch every step, no
            # data-dependent control flow.
            static_graph=True,
        )
    # Deterministic, contiguous-free shard: rank i takes every world-th pair.
    local_pairs = pairs[rank::world] if world > 1 else pairs
    if not local_pairs:
        raise SystemExit(
            f"[lora_fit] rank {rank} got no pairs ({len(pairs)} pairs < {world} ranks) "
            "— the launcher must cap num_processes at the pair count"
        )
    print(f"[lora_fit] {len(pairs)} pairs ({len(local_pairs)} on this rank), "
          f"{n_trainable/1e6:.1f}M trainable params")

    optimizer = torch.optim.AdamW(trainable, lr=float(params["lr"]))
    mb = max(1, int(params["micro_batch_size"]))
    # Shifted-label count: each pair contributes len(response) - prompt_len
    # response predictions (the CHOSEN response for DPO). Counted over ALL pairs
    # (global) so every rank normalizes by the same denominator. Constant across
    # steps (full batch). DPO's own loss is normalized by the global PAIR count.
    resp_key = "chosen_ids" if objective == "dpo" else "input_ids"
    total_resp_tokens = sum(len(p[resp_key]) - p["prompt_len"] for p in pairs)
    if total_resp_tokens <= 0:
        raise SystemExit("[lora_fit] pairs contain no response tokens")
    n_pairs_global = len(pairs)
    beta = float(params["beta"])
    sft_weight = float(params["sft_weight"])

    # Every rank must run the SAME number of micro-batches and synchronize at
    # the SAME indices — otherwise the ranks enqueue different collectives and
    # NCCL deadlocks. pairs[rank::world] hands rank r ceil((N-r)/world) pairs,
    # so the global maximum is ceil(N/world), which every rank can compute
    # without communicating. Ranks that run out do a ZERO-WEIGHTED pass: the
    # forward/backward still fires the same reductions, but contributes no
    # gradient.
    max_local = -(-len(pairs) // world) if world > 1 else len(local_pairs)
    n_micro = -(-max_local // mb)
    # Collectives per step. sync_every=0 syncs once at the end of the step,
    # which means the gap between collectives is the WHOLE shard: 313 pairs is
    # 264 s on 2 GPUs, 1500 pairs is 1264 s — past NCCL's 600 s watchdog, so a
    # perfectly healthy large fit would abort. Syncing every K micro-batches
    # bounds that gap at any batch size. The update is unchanged: each
    # all-reduce averages the gradient accumulated so far, and averaging the
    # running sums yields the same total average as one sync at the end.
    sync_every = int(params.get("sync_every", 0) or 0)

    # Per-step loss to wandb, same convention as train.py / lora_rl.py: this
    # subprocess's OWN run, grouped under run.name. Rank 0 only. Never let a
    # wandb hiccup take down a fit that costs GPU-hours.
    wb = None
    if args.wandb_json and rank == 0:
        wb_cfg = json.loads(args.wandb_json)
        if wb_cfg.get("mode") != "disabled":
            try:
                import wandb

                os.environ["WANDB_MODE"] = wb_cfg["mode"]
                wb = wandb.init(
                    project=wb_cfg["project"], entity=wb_cfg.get("entity"),
                    mode=wb_cfg["mode"], group=wb_cfg["group"], name=wb_cfg["name"],
                    job_type="lora_fit", dir=wb_cfg.get("dir") or str(out_dir.parent),
                    config={"n_pairs": len(pairs), "world_size": world, **params},
                )
            except Exception as e:                        # noqa: BLE001
                print(f"[lora_fit] wandb disabled ({type(e).__name__}: {e})", flush=True)

    def micro_chunks():
        """(chunk, pad) per micro-batch; pad chunks are fillers that every rank
        runs zero-weighted so DDP collective counts stay equal."""
        for mi in range(n_micro):
            chunk = local_pairs[mi * mb:(mi + 1) * mb]
            pad = not chunk
            if pad:
                chunk = local_pairs[:mb]
            yield mi, chunk, pad

    def dpo_seqs(chunk):
        return ([(p["chosen_ids"], p["prompt_len"]) for p in chunk]
                + [(p["rejected_ids"], p["prompt_len"]) for p in chunk])

    # DPO reference logprobs: ONE no-grad pass over the local pairs from the
    # fit's starting weights ("init" = the adapter that produced the rejected
    # samples) or with the adapter disabled ("base"). Precomputing them means
    # the reference policy never has to be held as a second model.
    ref_logp: dict[int, tuple] = {}
    if objective == "dpo":
        import contextlib

        ref_ctx = (peft_model.disable_adapter() if params["reference"] == "base"
                   else contextlib.nullcontext())
        with torch.no_grad(), ref_ctx:
            for mi, chunk, pad in micro_chunks():
                if pad:
                    continue
                ids, mask, labels = _pack(dpo_seqs(chunk), device)
                nll, _ = _seq_nll(peft_model(input_ids=ids, attention_mask=mask).logits, labels)
                half = len(chunk)
                for i in range(half):
                    ref_logp[mi * mb + i] = (-float(nll[i]), -float(nll[half + i]))
        if rank == 0:
            print(f"[lora_fit] dpo reference ({params['reference']}) logprobs cached "
                  f"for {len(ref_logp)} local pairs")

    loss_per_step: list[float] = []
    margin_per_step: list[float] = []
    acc_per_step: list[float] = []
    for step in range(int(params["steps"])):
        optimizer.zero_grad(set_to_none=True)
        step_ce = 0.0          # SFT: summed response NLL (reported per token)
        step_dpo = 0.0         # DPO: summed pair loss (reported per pair)
        step_margin = 0.0
        step_acc = 0.0
        for mi, chunk, pad in micro_chunks():
            if objective == "sft":
                ids, mask, labels = _pack([(p["input_ids"], p["prompt_len"]) for p in chunk], device)
                logits = model(input_ids=ids, attention_mask=mask).logits
                nll, _ = _seq_nll(logits, labels)
                ce = nll.sum()
                # Normalize by the FULL batch's response-token count so micro-
                # batch gradients sum to the full-batch mean regardless of mb
                # topology. Under DDP the all-reduce AVERAGES gradients across
                # ranks, so the local loss carries a world_size factor to undo
                # that averaging — the update then matches the single-process
                # full-batch one.
                loss = ce * (0.0 if pad else world / total_resp_tokens)
            else:
                ids, mask, labels = _pack(dpo_seqs(chunk), device)
                logits = model(input_ids=ids, attention_mask=mask).logits
                nll, _ = _seq_nll(logits, labels)
                half = len(chunk)
                pol_c, pol_r = -nll[:half], -nll[half:]
                # filler micro-batches (DDP rank with fewer pairs) have no cached
                # reference — they are zero-weighted below, so any value works
                ref = ([(0.0, 0.0)] * half if pad
                       else [ref_logp[mi * mb + i] for i in range(half)])
                ref_c = torch.tensor([r[0] for r in ref], device=device)
                ref_r = torch.tensor([r[1] for r in ref], device=device)
                margin = beta * ((pol_c - ref_c) - (pol_r - ref_r))
                dpo = -F.logsigmoid(margin).sum()
                # pair-normalized preference loss (+ optional RPO NLL on chosen,
                # token-normalized like the SFT path); same world factor.
                loss = dpo * (0.0 if pad else world / n_pairs_global)
                nll_c = nll[:half].sum()
                if sft_weight > 0:
                    loss = loss + nll_c * (
                        0.0 if pad else sft_weight * world / total_resp_tokens)
                if not pad:
                    # reported loss = the optimized objective per pair (DPO term
                    # + the RPO NLL term rescaled to the same per-pair unit)
                    step_dpo += float(dpo.detach()) + (
                        sft_weight * float(nll_c.detach()) * n_pairs_global / total_resp_tokens)
                    step_margin += float(margin.detach().sum())
                    step_acc += float((margin.detach() > 0).sum())
            sync = mi == n_micro - 1 or (sync_every and (mi + 1) % sync_every == 0)
            if world > 1 and not sync:
                with model.no_sync():   # accumulate without communicating
                    loss.backward()
            else:
                loss.backward()
            if objective == "sft" and not pad:
                step_ce += float(ce.detach())
        if float(params["max_grad_norm"]) > 0:
            torch.nn.utils.clip_grad_norm_(trainable, float(params["max_grad_norm"]))
        optimizer.step()
        if objective == "sft":
            stats_t = [step_ce / total_resp_tokens, 0.0, 0.0]
        else:
            stats_t = [step_dpo / n_pairs_global, step_margin / n_pairs_global,
                       step_acc / n_pairs_global]
        if world > 1:  # report the GLOBAL numbers, not this rank's shard
            t = torch.tensor(stats_t, device=device)
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
            stats_t = [float(v) for v in t.tolist()]
        step_loss, step_m, step_a = stats_t
        loss_per_step.append(step_loss)
        margin_per_step.append(step_m)
        acc_per_step.append(step_a)
        if rank == 0:
            extra = (f" margin {step_m:.4f} pref_acc {step_a:.3f}"
                     if objective == "dpo" else "")
            print(f"[lora_fit] step {step + 1}/{params['steps']} loss {step_loss:.4f}{extra}")
            if wb is not None:
                log = {"lora_fit_loss": step_loss}
                if objective == "dpo":
                    log.update(reward_margin=step_m, pref_acc=step_a)
                wb.log(log, step=step + 1)

    if world > 1:
        torch.distributed.barrier()
    if rank != 0:  # every rank holds the same LoRA weights; rank 0 persists them
        torch.distributed.destroy_process_group()
        return
    peft_model.save_pretrained(str(out_dir))
    write_json(out_dir / "fit_meta.json", {
        "base_model": args.model,
        "init_adapter": args.init_adapter or None,
        "n_pairs": len(pairs),
        "world_size": world,
        "total_resp_tokens": total_resp_tokens,
        "loss_per_step": [round(v, 6) for v in loss_per_step],
        "objective": objective,
        **({"reward_margin_per_step": [round(v, 6) for v in margin_per_step],
            "pref_acc_per_step": [round(v, 6) for v in acc_per_step]}
           if objective == "dpo" else {}),
        "seed": int(params["seed"]),
        "seconds": round(time.time() - t0, 1),
        "params": params,
        # bf16 GPU kernels are not bitwise deterministic across devices (nor is
        # DDP gradient reduction vs a single process); the .done marker +
        # content-addressed dir make reruns no-ops instead.
    })
    mark_done(out_dir / "adapter_config.json", count=len(pairs), config_hash=args.cache_key)
    if wb is not None:
        wb.summary["lora_fit_loss_first"] = loss_per_step[0]
        wb.summary["lora_fit_loss_last"] = loss_per_step[-1]
        wb.finish()
    print(f"[lora_fit] saved adapter -> {out_dir} ({time.time() - t0:.0f}s)")
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main(sys.argv[1:])

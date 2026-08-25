"""lora_fit round-trip on a tiny random model — CPU only (slow)."""

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
from pathlib import Path

from expert_iter.config import Config


def _tiny_base(tmp_path):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    base = tmp_path / "base"
    LlamaForCausalLM(LlamaConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=64,
    )).save_pretrained(base)
    return base


FIT_PARAMS = {
    "r": 4, "lora_alpha": 8, "lr": 1e-3, "steps": 2, "seed": 0,
    "bf16": False, "gradient_checkpointing": False, "attn_implementation": "eager",
    "target_modules": ["q_proj", "v_proj"],
    "micro_batch_size": 1, "max_grad_norm": 0.0, "dropout": 0.0,
}


@pytest.mark.slow
def test_ddp_fit_matches_single_process(tmp_path):
    """The world_size factor must undo DDP's gradient averaging: a 2-rank fit
    (gloo/CPU) must land on the same LoRA weights as the single-process
    full-batch fit. Guards the same compensation trap as api_notes finding 7."""
    import torch
    from safetensors.torch import load_file

    from expert_iter.utils import write_jsonl

    base = _tiny_base(tmp_path)
    pairs_path = tmp_path / "pairs.jsonl"
    write_jsonl(pairs_path, [
        {"qid": "a", "input_ids": [1, 2, 3, 4, 5, 6], "prompt_len": 2},
        {"qid": "b", "input_ids": [5, 4, 3, 2, 7], "prompt_len": 1},
        {"qid": "c", "input_ids": [9, 8, 7, 6, 5, 4, 3], "prompt_len": 3},
        {"qid": "d", "input_ids": [2, 3, 4, 5], "prompt_len": 1},
    ])

    def run(out, argv_prefix, extra_env=None):
        # argv_prefix ends with the flag that introduces the module name
        cmd = [*argv_prefix, "expert_iter.lora_fit",
               "--model", str(base), "--pairs", str(pairs_path), "--out", str(out),
               "--params-json", json.dumps(FIT_PARAMS), "--cache-key", out.name]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", **(extra_env or {})}
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
        return load_file(out / "adapter_model.safetensors")

    single = run(tmp_path / "single", [sys.executable, "-m"])
    # torchrun, not `accelerate launch --cpu`: the CPU launcher goes through MPI
    # and leaves RANK/WORLD_SIZE unset (so the fit would silently run on one
    # rank). On GPU, `accelerate launch` does set them — that is the production
    # path (verified by scripts/check_env.py's DDP probe).
    ddp = run(tmp_path / "ddp",
              [sys.executable, "-m", "torch.distributed.run",
               "--nproc_per_node", "2", "--master_port", "29577", "-m"],
              {"OMP_NUM_THREADS": "1"})

    assert set(single) == set(ddp)
    for k in single:
        assert torch.allclose(single[k], ddp[k], atol=1e-5, rtol=1e-4), k
    meta = json.loads((tmp_path / "ddp" / "fit_meta.json").read_text())
    assert meta["world_size"] == 2 and meta["n_pairs"] == 4


@pytest.mark.slow
def test_lora_fit_roundtrip_and_resume(tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)  # force CPU

    from transformers import LlamaConfig, LlamaForCausalLM

    from expert_iter import lora_fit
    from expert_iter.utils import write_jsonl

    torch.manual_seed(0)
    base = tmp_path / "base"
    LlamaForCausalLM(LlamaConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=64,
    )).save_pretrained(base)

    pairs_path = tmp_path / "pairs.jsonl"
    write_jsonl(pairs_path, [
        {"qid": "a", "input_ids": [1, 2, 3, 4, 5], "prompt_len": 2},
        {"qid": "b", "input_ids": [5, 4, 3, 2], "prompt_len": 1},
    ])
    out = tmp_path / "adapter"
    params = {
        "r": 4, "lora_alpha": 8, "lr": 1e-3, "steps": 2, "seed": 0,
        "bf16": False, "gradient_checkpointing": False,
        "attn_implementation": "eager",
        "target_modules": ["q_proj", "v_proj"],
        "micro_batch_size": 1, "max_grad_norm": 1.0, "dropout": 0.0,
    }
    argv = ["--model", str(base), "--pairs", str(pairs_path), "--out", str(out),
            "--params-json", json.dumps(params), "--cache-key", "k1"]
    lora_fit.main(argv)

    assert (out / "adapter_config.json").exists()
    assert list(out.glob("adapter_model*")), "no adapter weights saved"
    meta = json.loads((out / "fit_meta.json").read_text())
    assert meta["n_pairs"] == 2 and len(meta["loss_per_step"]) == 2
    assert meta["total_resp_tokens"] == 3 + 3

    # resume: matching .done marker skips the refit (weights untouched)
    weights = next(out.glob("adapter_model*"))
    mtime = weights.stat().st_mtime_ns
    lora_fit.main(argv)
    assert weights.stat().st_mtime_ns == mtime

    # warm start (adaptive tau_E rounds): --init-adapter loads the previous
    # round's weights and records the chain in fit_meta
    out2 = tmp_path / "adapter_r2"
    lora_fit.main(["--model", str(base), "--pairs", str(pairs_path),
                   "--out", str(out2), "--params-json", json.dumps(params),
                   "--cache-key", "k2", "--init-adapter", str(out)])
    meta2 = json.loads((out2 / "fit_meta.json").read_text())
    assert meta2["init_adapter"] == str(out)
    assert list(out2.glob("adapter_model*"))

    # LoRA shape comes from the saved adapter; a params mismatch must fail loudly
    bad = {**params, "r": 8}
    with pytest.raises(SystemExit, match="init-adapter r="):
        lora_fit.main(["--model", str(base), "--pairs", str(pairs_path),
                       "--out", str(tmp_path / "bad"), "--params-json", json.dumps(bad),
                       "--cache-key", "k3", "--init-adapter", str(out)])


# ---------------------------------------------------------------------------
# objective=dpo (staged stage2_objective): preference pairs, precomputed reference
# ---------------------------------------------------------------------------

DPO_PAIRS = [
    {"qid": "a", "prompt_len": 2, "chosen_ids": [1, 2, 3, 4, 5, 6], "rejected_ids": [1, 2, 9, 8, 7]},
    {"qid": "b", "prompt_len": 1, "chosen_ids": [5, 4, 3, 2], "rejected_ids": [5, 6, 7, 8, 9]},
    {"qid": "c", "prompt_len": 3, "chosen_ids": [9, 8, 7, 6, 5], "rejected_ids": [9, 8, 7, 1, 2, 3]},
]


def _run_dpo(tmp_path, out_name, pairs, **overrides):
    import json as _json
    from expert_iter import lora_fit
    from expert_iter.utils import write_jsonl

    base = tmp_path / "base"
    if not base.exists():
        _tiny_base(tmp_path)
    pairs_path = tmp_path / f"{out_name}_pairs.jsonl"
    write_jsonl(pairs_path, pairs)
    out = tmp_path / out_name
    params = {**FIT_PARAMS, "objective": "dpo", "beta": 0.5, "steps": 4, "lr": 5e-3,
              **overrides}
    lora_fit.main(["--model", str(base), "--pairs", str(pairs_path), "--out", str(out),
                   "--params-json", _json.dumps(params), "--cache-key", out_name])
    return out, _json.loads((out / "fit_meta.json").read_text())


@pytest.mark.slow
def test_dpo_roundtrip_margin_rises_and_resume(tmp_path, monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    out, meta = _run_dpo(tmp_path, "dpo", DPO_PAIRS)
    assert list(out.glob("adapter_model*")), "no adapter weights saved"
    assert meta["objective"] == "dpo" and meta["n_pairs"] == 3
    assert len(meta["loss_per_step"]) == 4 == len(meta["reward_margin_per_step"])
    # chosen response tokens only (4 + 3 + 2), the SFT-style normalizer
    assert meta["total_resp_tokens"] == 4 + 3 + 2
    # the objective does what it says: chosen gains on rejected relative to the
    # reference, monotonically over 4 steps, and the loss falls
    m = meta["reward_margin_per_step"]
    assert m[0] == pytest.approx(0.0, abs=1e-4)       # step 1 == reference policy
    assert m[-1] > m[0] and meta["loss_per_step"][-1] < meta["loss_per_step"][0]
    assert 0.0 <= meta["pref_acc_per_step"][-1] <= 1.0
    # resume: matching .done marker skips the refit
    from expert_iter import lora_fit
    import json as _json
    weights = next(out.glob("adapter_model*"))
    mtime = weights.stat().st_mtime_ns
    lora_fit.main(["--model", str(tmp_path / "base"), "--pairs", str(tmp_path / "dpo_pairs.jsonl"),
                   "--out", str(out), "--params-json",
                   _json.dumps({**FIT_PARAMS, "objective": "dpo", "beta": 0.5, "steps": 4, "lr": 5e-3}),
                   "--cache-key", "dpo"])
    assert weights.stat().st_mtime_ns == mtime


@pytest.mark.slow
def test_dpo_loss_invariant_to_micro_batch_size(tmp_path, monkeypatch):
    """Pair normalization: the per-step loss must not depend on how the pairs
    are split into micro-batches (same property tests/test_loss_invariance.py
    guards for SFT)."""
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    _, m1 = _run_dpo(tmp_path, "mb1", DPO_PAIRS, micro_batch_size=1, steps=2)
    _, m2 = _run_dpo(tmp_path, "mb2", DPO_PAIRS, micro_batch_size=2, steps=2)
    for a, b in zip(m1["loss_per_step"], m2["loss_per_step"]):
        assert a == pytest.approx(b, rel=1e-4, abs=1e-6)
    for a, b in zip(m1["reward_margin_per_step"], m2["reward_margin_per_step"]):
        assert a == pytest.approx(b, rel=1e-4, abs=1e-6)


@pytest.mark.slow
def test_dpo_sft_weight_and_base_reference(tmp_path, monkeypatch):
    """sft_weight adds the chosen NLL term (loss strictly larger at step 1 for
    the same weights); reference=base runs with the adapter disabled."""
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    _, plain = _run_dpo(tmp_path, "plain", DPO_PAIRS, steps=1)
    _, rpo = _run_dpo(tmp_path, "rpo", DPO_PAIRS, steps=1, sft_weight=1.0)
    assert rpo["loss_per_step"][0] > plain["loss_per_step"][0]
    _, base_ref = _run_dpo(tmp_path, "baseref", DPO_PAIRS, steps=2, reference="base")
    assert base_ref["params"]["reference"] == "base"
    assert len(base_ref["reward_margin_per_step"]) == 2


@pytest.mark.slow
def test_ddp_dpo_uneven_pairs_matches_single_process(tmp_path):
    """3 DPO pairs over 2 ranks: rank 1 runs a zero-weighted filler micro-batch
    that has no cached reference logprob (the KeyError that killed the first
    4B DPO arm), and the DDP weights must still equal the single-process fit."""
    import torch
    from safetensors.torch import load_file

    from expert_iter.utils import write_jsonl

    base = _tiny_base(tmp_path)
    pairs_path = tmp_path / "dpo_pairs.jsonl"
    write_jsonl(pairs_path, DPO_PAIRS)
    params = {**FIT_PARAMS, "objective": "dpo", "beta": 0.5, "steps": 2, "lr": 5e-3,
              "sft_weight": 0.5}

    def run(out, argv_prefix, extra_env=None):
        cmd = [*argv_prefix, "expert_iter.lora_fit",
               "--model", str(base), "--pairs", str(pairs_path), "--out", str(out),
               "--params-json", json.dumps(params), "--cache-key", out.name]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", **(extra_env or {})}
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
        return load_file(out / "adapter_model.safetensors")

    single = run(tmp_path / "single", [sys.executable, "-m"])
    ddp = run(tmp_path / "ddp",
              [sys.executable, "-m", "torch.distributed.run",
               "--nproc_per_node", "2", "--master_port", "29578", "-m"],
              {"OMP_NUM_THREADS": "1"})
    for k in single:
        assert torch.allclose(single[k], ddp[k], atol=1e-5, rtol=1e-4), k
    m1 = json.loads((tmp_path / "single" / "fit_meta.json").read_text())
    m2 = json.loads((tmp_path / "ddp" / "fit_meta.json").read_text())
    assert m2["world_size"] == 2
    for a, b in zip(m1["reward_margin_per_step"], m2["reward_margin_per_step"]):
        assert a == pytest.approx(b, rel=1e-3, abs=1e-5)


def test_dpo_rejects_sft_schema(tmp_path, monkeypatch):
    import json as _json
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    from expert_iter import lora_fit
    from expert_iter.utils import write_jsonl

    pairs_path = tmp_path / "sft_pairs.jsonl"
    write_jsonl(pairs_path, [{"qid": "a", "input_ids": [1, 2, 3], "prompt_len": 1}])
    with pytest.raises(SystemExit, match="needs pair keys"):
        lora_fit.main(["--model", "unused", "--pairs", str(pairs_path),
                       "--out", str(tmp_path / "x"),
                       "--params-json", _json.dumps({"objective": "dpo"}),
                       "--cache-key", "x"])


# ---------------------------------------------------------------------------
# wandb reporting for the transient fit
# ---------------------------------------------------------------------------

def test_wandb_params_shape():
    """Same convention as train.py / lora_rl.py: the subprocess gets its own run
    grouped under run.name."""
    from expert_iter.lora_sft import wandb_params

    cfg = Config.load(None, overrides=["run.name=bridge_toy_cliff",
                                       "run.wandb.mode=offline"])
    wb = wandb_params(cfg, run_name="bridge_toy_cliff/iter0/fit_pooled_c0")
    assert wb == {"project": "ei_reasoning", "entity": None, "mode": "offline",
                  "group": "bridge_toy_cliff",
                  "name": "bridge_toy_cliff/iter0/fit_pooled_c0"}


def test_wandb_settings_do_not_change_the_fit_key(monkeypatch, tmp_path):
    """Where the loss curve is reported must never invalidate a cached adapter."""
    from expert_iter import lora_sft as mod

    seen = []

    def fake_run(cmd, **kw):
        seen.append(cmd)
        out = Path(cmd[cmd.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "adapter_config.json").write_text("{}")
        from expert_iter.utils import mark_done
        mark_done(out / "adapter_config.json", count=1,
                  config_hash=cmd[cmd.index("--cache-key") + 1])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    monkeypatch.setattr(mod, "visible_gpus", lambda _: [0])
    monkeypatch.setattr(mod, "_model_fingerprint", lambda _: "fp")
    cfg = Config.load(None, overrides=["engine.enable_lora=true",
                                       "improve.operator=lora_sft"])
    pairs = [{"qid": "q1", "input_ids": [1, 2, 3], "prompt_len": 1}]
    params = {"r": 16, "steps": 1}

    wb = {"project": "p", "mode": "offline", "group": "g", "name": "n", "entity": None}
    d1, _ = mod._fit_adapter("m", pairs, tmp_path / "a", "c", params, cfg)
    d2, _ = mod._fit_adapter("m", pairs, tmp_path / "a", "c", params, cfg, wandb=wb)
    assert d1 == d2                       # same content-addressed adapter dir
    assert len(seen) == 1                 # the second call was a pure cache hit
    assert "--wandb-json" not in seen[0]

    # ... and when a fit DOES run, the settings reach the subprocess
    d3, _ = mod._fit_adapter("m", pairs, tmp_path / "b", "c", params, cfg, wandb=wb)
    assert len(seen) == 2 and "--wandb-json" in seen[1]
    assert json.loads(seen[1][seen[1].index("--wandb-json") + 1])["name"] == "n"
    assert d3.name == d1.name             # same fit key, different adapters root


def test_lora_fit_accepts_wandb_json():
    """The CLI must take the flag even when nothing else changes."""
    import expert_iter.lora_fit as lf

    ap = lf.main.__globals__["argparse"].ArgumentParser()
    # cheapest check that does not run a fit: the module parses its own argv
    import subprocess as sp
    r = sp.run([sys.executable, "-m", "expert_iter.lora_fit", "--help"],
               capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1]))
    assert "--wandb-json" in r.stdout

"""Single nested config: YAML <-> dataclasses, dot-path CLI overrides, frozen snapshot.

Usage from any stage:
    cfg = Config.load("configs/ei_default.yaml", overrides=["rollout.n=4"])
Unknown YAML keys are hard errors (typos must not silently no-op).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import get_args, get_origin, get_type_hints

import yaml

from .utils import write_json


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

@dataclass
class WandbCfg:
    project: str = "ei_reasoning"
    entity: str | None = None
    mode: str = "online"          # online | offline | disabled


@dataclass
class RunCfg:
    name: str = "ei_dev"
    output_root: str = "runs"
    seed: int = 17
    wandb: WandbCfg = field(default_factory=WandbCfg)


@dataclass
class ModelCfg:
    base: str = "Qwen/Qwen3-4B-Instruct-2507"
    dtype: str = "bfloat16"
    system_prompt: str | None = None
    # Forwarded to apply_chat_template by ALL prompt-rendering stages. Hybrid
    # Qwen3 (0.6B/4B/8B) takes {enable_thinking: true|false}; Qwen3-*-2507
    # Instruct and R1-Distill templates take no switches — leave empty.
    chat_template_kwargs: dict = field(default_factory=dict)


@dataclass
class EngineCfg:
    gpus: list[int] | None = None      # null -> CUDA_VISIBLE_DEVICES -> device_count
    tensor_parallel: int = 1           # >1 groups GPUs per vLLM worker
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 12288         # must cover the longest scored sequence
    score_batch_size: int = 256
    enable_prefix_caching: bool = True
    enforce_eager: bool = False


@dataclass
class DataCfg:
    adapter: str = "openthoughts_math"
    adapter_args: dict = field(default_factory=dict)
    # Appended to every question when rendering the user turn; keeps grading
    # reliable by pinning the final-answer format.
    question_suffix: str = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
    eval_holdout: int = 200            # split off by stable qid hash, fixed across iterations
    accumulate: bool = True            # STaR-style union of all iterations' filtered data


@dataclass
class RolloutCfg:
    n: int = 8
    temperature: float = 1.0
    top_p: float = 0.95
    max_tokens: int = 8192
    capture_logprobs: bool = False


@dataclass
class PartitionCfg:
    verifier: str = "math"
    solved_keep_max: int = 4           # cap correct trajectories kept per question
    solved_selection: str = "shortest"  # shortest | first | random


@dataclass
class AnchorCfg:
    policy: str = "fixed_fraction"     # ⚗ extension point I
    params: dict = field(default_factory=lambda: {"fraction": 0.3, "min_tokens": 32, "max_tokens": 2048})
    base_selection: str = "first_failed"  # first_failed | longest | random


@dataclass
class TeacherCfg:
    model: str | None = None
    gpus: list[int] | None = None
    quantization: str | None = None


@dataclass
class ImproveCfg:
    operator: str = "self_resample"    # ⚗ extension point II
    n: int = 8                         # continuations sampled per anchor
    temperature: float = 1.2
    top_p: float = 0.98
    max_tokens: int = 8192
    rounds: int = 1                    # ⚗ multi-round retry budget for future operators
    teacher: TeacherCfg = field(default_factory=TeacherCfg)


@dataclass
class LogprobGateCfg:
    enabled: bool = False
    min_mean_logprob: float = -1.5
    scope: str = "continuation"        # continuation | full


@dataclass
class FilterCfg:
    gates: list[str] = field(default_factory=lambda: ["correctness", "no_external_context", "length", "dedup"])
    max_total_tokens: int = 10240      # prompt+anchor+continuation cap (must fit train.max_seq_len)
    max_per_question: int = 2          # quota of improved trajectories kept per qid
    logprob_gate: LogprobGateCfg = field(default_factory=LogprobGateCfg)


@dataclass
class SftCfg:
    lr: float = 1.0e-5
    epochs: float = 2.0
    scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    global_batch_size: int = 32
    micro_batch_size: int = 1
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    bf16: bool = True
    gradient_checkpointing: bool = True
    packing: bool = False              # errors if true with non-uniform region_weights
    logging_steps: int = 5
    # ⚗ extension point IV: per-region loss weights.
    # "solution" = the response region of natively-solved examples.
    region_weights: dict = field(default_factory=lambda: {
        "prompt": 0.0, "anchor": 0.0, "continuation": 1.0, "solution": 1.0,
    })


@dataclass
class DpoCfg:
    lr: float = 5.0e-7
    beta: float = 0.1
    epochs: float = 1.0
    global_batch_size: int = 16
    micro_batch_size: int = 1
    loss_type: str = "sigmoid"
    max_grad_norm: float = 1.0
    logging_steps: int = 5


@dataclass
class TrainCfg:
    objective: str = "sft"             # sft | dpo | sft+dpo
    init_from: str = "base"            # base (STaR-style) | last
    backend: str = "zero2"             # single | zero2 | zero3 | fsdp2
    max_seq_len: int = 10240
    sft: SftCfg = field(default_factory=SftCfg)
    dpo: DpoCfg = field(default_factory=DpoCfg)


@dataclass
class LoopCfg:
    iterations: int = 4
    stages: list[str] = field(default_factory=lambda: [
        "rollout", "partition", "anchor", "improve", "filters",
        "build_dataset", "train", "eval", "benchmark_eval",
    ])


@dataclass
class PasskCfg:
    k: int = 8
    temperature: float = 0.7
    top_p: float = 0.95


@dataclass
class BenchmarkCfg:
    """One external benchmark for the benchmark_eval stage.

    `name` alone selects a preset from data.BENCHMARK_PRESETS (aime24, aime25,
    aime26, hmmt25, math500, math500_hard); adapter/adapter_args override or
    extend it. Sampling knobs follow Qwen3 best practices per model mode —
    non-thinking: temp 0.7 / top_p 0.8 / top_k 20; thinking: 0.6 / 0.95 / 20.
    n=1 with temperature 0.0 gives greedy pass@1.
    """

    name: str = ""
    adapter: str = ""                  # "" -> preset lookup by name
    adapter_args: dict = field(default_factory=dict)
    n: int = 8                         # samples per problem -> pass@n / avg@n / maj@n
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20                    # -1 disables
    min_p: float = 0.0
    max_tokens: int = 16384


@dataclass
class EvalCfg:
    greedy_pass1: bool = True
    passk: PasskCfg = field(default_factory=PasskCfg)
    max_tokens: int = 8192
    # ---- benchmark_eval stage ----
    benchmarks: list[BenchmarkCfg] = field(default_factory=list)
    benchmark_every: int = 1           # run benchmarks every N iterations (others skip)
    benchmark_verifier: str = "math_strict"


@dataclass
class Config:
    run: RunCfg = field(default_factory=RunCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    engine: EngineCfg = field(default_factory=EngineCfg)
    data: DataCfg = field(default_factory=DataCfg)
    rollout: RolloutCfg = field(default_factory=RolloutCfg)
    partition: PartitionCfg = field(default_factory=PartitionCfg)
    anchor: AnchorCfg = field(default_factory=AnchorCfg)
    improve: ImproveCfg = field(default_factory=ImproveCfg)
    filter: FilterCfg = field(default_factory=FilterCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    loop: LoopCfg = field(default_factory=LoopCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)

    # -- construction -------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path | None, overrides: list[str] | None = None) -> "Config":
        raw: dict = {}
        if path is not None:
            raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        for ov in overrides or []:
            if "=" not in ov:
                raise ValueError(f"override must be key.path=value, got {ov!r}")
            key, value = ov.split("=", 1)
            _set_dotted(raw, key.strip(), yaml.safe_load(value))
        cfg = _from_dict(cls, raw, path="")
        cfg.validate()
        return cfg

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True).encode()
        ).hexdigest()[:16]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")

    # -- validation ---------------------------------------------------------

    def validate(self) -> None:
        t = self.train
        if t.objective not in ("sft", "dpo", "sft+dpo"):
            raise ValueError(f"train.objective: {t.objective!r}")
        if t.init_from not in ("base", "last"):
            raise ValueError(f"train.init_from: {t.init_from!r}")
        if t.backend not in ("single", "zero2", "zero3", "fsdp2"):
            raise ValueError(f"train.backend: {t.backend!r}")
        weights = set(t.sft.region_weights.values())
        if t.sft.packing and weights != {1.0}:
            raise ValueError(
                "train.sft.packing=true with non-uniform region_weights is not "
                "supported: the packed-order weight splicing is not implemented. "
                "Disable packing or set all region weights to 1.0."
            )
        if self.filter.max_total_tokens > t.max_seq_len:
            raise ValueError(
                f"filter.max_total_tokens ({self.filter.max_total_tokens}) exceeds "
                f"train.max_seq_len ({t.max_seq_len}): filtered examples would be truncated."
            )
        if self.engine.max_model_len < t.max_seq_len:
            raise ValueError(
                f"engine.max_model_len ({self.engine.max_model_len}) < train.max_seq_len "
                f"({t.max_seq_len}): logprob scoring of full trajectories would fail."
            )
        if self.partition.solved_selection not in ("shortest", "first", "random"):
            raise ValueError(f"partition.solved_selection: {self.partition.solved_selection!r}")
        if self.eval.benchmark_every < 1:
            raise ValueError(f"eval.benchmark_every must be >= 1, got {self.eval.benchmark_every}")
        seen_bench = set()
        for i, b in enumerate(self.eval.benchmarks):
            if not b.name:
                raise ValueError(f"eval.benchmarks[{i}]: name is required")
            if b.name in seen_bench:
                raise ValueError(f"eval.benchmarks: duplicate name {b.name!r}")
            seen_bench.add(b.name)
            if b.n < 1:
                raise ValueError(f"eval.benchmarks[{i}] ({b.name}): n must be >= 1")


# ---------------------------------------------------------------------------
# dict -> dataclass with unknown-key errors
# ---------------------------------------------------------------------------

def _from_dict(cls, raw: dict, path: str):
    if not isinstance(raw, dict):
        raise TypeError(f"config section {path or '<root>'} must be a mapping, got {type(raw).__name__}")
    hints = get_type_hints(cls)
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(raw) - set(known)
    if unknown:
        raise KeyError(f"unknown config key(s) {sorted(unknown)} under {path or '<root>'}")
    kwargs = {}
    for name, f in known.items():
        if name not in raw:
            continue
        value = raw[name]
        target = hints[f.name]
        # list[Dataclass] fields (e.g. eval.benchmarks): map each element.
        if get_origin(target) is list:
            (elem,) = get_args(target)
            if dataclasses.is_dataclass(elem) and isinstance(value, list):
                kwargs[name] = [
                    _from_dict(elem, v, f"{path}.{name}[{i}]" if path else f"{name}[{i}]")
                    for i, v in enumerate(value)
                ]
                continue
        # Unwrap Optional[X] / unions: pick the sole dataclass member if any.
        union_args = [a for a in get_args(target) if dataclasses.is_dataclass(a)]
        if union_args:
            target = union_args[0]
        if dataclasses.is_dataclass(target) and isinstance(value, dict):
            kwargs[name] = _from_dict(target, value, f"{path}.{name}" if path else name)
        else:
            kwargs[name] = _coerce_scalar(target, value, f"{path}.{name}" if path else name)
    return cls(**kwargs)


def _coerce_scalar(target, value, path: str):
    """YAML 1.1 parses '2e-5' (no decimal point) as a STRING; coerce scalars to
    the field's declared type so numeric overrides always behave."""
    try:
        if target is float and isinstance(value, (int, str)):
            return float(value)
        if target is int and isinstance(value, str):
            return int(value)
    except ValueError as e:
        raise ValueError(f"config value {path}={value!r} is not a valid {target.__name__}") from e
    return value


def _set_dotted(d: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
        if not isinstance(cur, dict):
            raise ValueError(f"override path {dotted!r} collides with non-mapping value")
    cur[keys[-1]] = value


# ---------------------------------------------------------------------------
# Shared stage CLI
# ---------------------------------------------------------------------------

def stage_argparser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", required=True, help="path to the YAML config")
    p.add_argument("--override", action="append", default=[], metavar="a.b=c",
                   help="dot-path config override (repeatable)")
    p.add_argument("--run-dir", required=True, help="runs/<name> directory")
    p.add_argument("--iter", type=int, required=True, dest="iteration")
    p.add_argument("--model-path", required=True,
                   help="HF hub id or local checkpoint dir of the CURRENT policy")
    return p


def load_stage_config(args: argparse.Namespace) -> Config:
    cfg = Config.load(args.config, overrides=args.override)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    frozen = run_dir / "config.yaml"
    if not frozen.exists():
        cfg.save(frozen)
        write_json(run_dir / "config.hash.json", {"config_hash": cfg.hash()})
    return cfg

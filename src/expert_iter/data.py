"""Dataset adapters -> canonical QuestionRecord list, plus the holdout split.

Adapters own all dataset-specific quirks (column names, answer normalization,
domain filtering). Downstream stages only ever see QuestionRecord.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .records import QuestionRecord
from .registry import ADAPTERS, build, register
from .utils import read_jsonl, stable_hash
from .verifier import last_boxed


class DatasetAdapter(ABC):
    name: str

    @abstractmethod
    def load(self, args: dict) -> list[QuestionRecord]:
        ...


@register(ADAPTERS, "openthoughts_math")
class OpenThoughtsMathAdapter(DatasetAdapter):
    """open-thoughts/OpenThoughts-114k, `metadata` config, math rows only.

    We use ONLY the problem text and the ground-truth final answer; the
    dataset's own R1 reasoning traces are never read — trajectories must come
    from our policy or improvement operator, not distillation.

    Rows are dropped when: domain != math, no ground truth, or math-verify
    cannot parse the gold answer (unverifiable => useless for EI).
    """

    def load(self, args: dict) -> list[QuestionRecord]:
        import datasets as hf_datasets

        from .verifier import MathVerifier

        hf_name = args.get("hf_name", "open-thoughts/OpenThoughts-114k")
        config = args.get("config", "metadata")
        split = args.get("split", "train")
        domain = args.get("domain", "math")
        max_items = args.get("max_items")

        ds = hf_datasets.load_dataset(hf_name, config, split=split)

        question_col = _first_present(ds.column_names, ["problem", "question", "prompt"])
        answer_col = _first_present(
            ds.column_names, ["ground_truth_solution", "solution", "answer", "ground_truth"]
        )
        if question_col is None or answer_col is None:
            raise KeyError(
                f"{hf_name}[{config}] columns {ds.column_names} lack a recognizable "
                "question/ground-truth pair; pass adapter_args.config or use the "
                "local_jsonl adapter."
            )
        domain_col = _first_present(ds.column_names, ["domain", "category", "source_type"])

        verifier = MathVerifier()
        records: list[QuestionRecord] = []
        n_seen = n_domain = n_no_gold = n_unparsable = 0
        for row in ds:
            n_seen += 1
            if domain_col and domain and str(row.get(domain_col, "")).lower() != domain:
                n_domain += 1
                continue
            question = (row.get(question_col) or "").strip()
            gold = (row.get(answer_col) or "").strip()
            if not question or not gold:
                n_no_gold += 1
                continue
            final_answer = _extract_final_answer(gold)
            if not verifier.gold_parsable(final_answer):
                n_unparsable += 1
                continue
            records.append(
                QuestionRecord(
                    qid="ot-" + stable_hash(question),
                    question=question,
                    final_answer=final_answer,
                    domain="math",
                    meta={"hf_name": hf_name, "row_source": row.get("source", "")},
                )
            )
            if max_items and len(records) >= max_items:
                break
        print(
            f"[data] openthoughts_math: kept {len(records)} / seen {n_seen} "
            f"(domain-filtered {n_domain}, no-gold {n_no_gold}, unparsable {n_unparsable})"
        )
        return _dedup_by_qid(records)


@register(ADAPTERS, "local_jsonl")
class LocalJsonlAdapter(DatasetAdapter):
    """Rows already in canonical form: {qid?, question, final_answer, domain?, meta?}."""

    def load(self, args: dict) -> list[QuestionRecord]:
        path = args["path"]
        max_items = args.get("max_items")
        records = []
        for row in read_jsonl(path):
            records.append(
                QuestionRecord(
                    qid=row.get("qid") or "local-" + stable_hash(row["question"]),
                    question=row["question"],
                    final_answer=str(row["final_answer"]),
                    domain=row.get("domain", "math"),
                    meta=row.get("meta", {}),
                )
            )
            if max_items and len(records) >= max_items:
                break
        return _dedup_by_qid(records)


# ---------------------------------------------------------------------------
# External benchmarks (benchmark_eval stage)
# ---------------------------------------------------------------------------

# Dataset ids/columns for aime24/25, hmmt25, math500 ported from OPSD's
# evaluate_math.py (community-standard sources, so numbers stay comparable);
# aime26 from MathArena. All six presets verified loadable 2026-08-06
# (30/30/30/30/500/134 questions). benchmark_eval reports (not raises) load
# failures, so a renamed/removed hub dataset can't kill an EI loop.
BENCHMARK_PRESETS: dict[str, dict] = {
    "aime24": {"hf_name": "HuggingFaceH4/aime_2024", "split": "train"},
    "aime25": {"hf_name": "yentinglin/aime_2025", "split": "train"},
    "aime26": {"hf_name": "MathArena/aime_2026", "split": "train"},
    "hmmt25": {"hf_name": "MathArena/hmmt_feb_2025", "split": "train"},
    "math500": {"hf_name": "HuggingFaceH4/MATH-500", "split": "test"},
    # "hard" = level-5 problems only (~134 of 500).
    "math500_hard": {"hf_name": "HuggingFaceH4/MATH-500", "split": "test", "min_level": 5},
}


@register(ADAPTERS, "hf_benchmark")
class HFBenchmarkAdapter(DatasetAdapter):
    """Generic HF competition-benchmark adapter.

    Unlike the training adapters, rows are NEVER dropped for unparsable gold —
    dropping would silently change the benchmark's denominator. Grading copes
    via math_strict's string-equality fallback.
    """

    def load(self, args: dict) -> list[QuestionRecord]:
        import datasets as hf_datasets

        hf_name = args["hf_name"]
        split = args.get("split", "test")
        bench_name = args.get("bench_name", hf_name.rsplit("/", 1)[-1].lower())
        min_level = args.get("min_level")
        max_items = args.get("max_items")

        ds = hf_datasets.load_dataset(hf_name, split=split)
        question_col = args.get("question_col") or _first_present(
            ds.column_names, ["problem", "question", "prompt"]
        )
        answer_col = args.get("answer_col") or _first_present(
            ds.column_names, ["answer", "final_answer", "solution"]
        )
        if question_col is None or answer_col is None:
            raise KeyError(
                f"{hf_name}[{split}] columns {ds.column_names} lack a recognizable "
                "question/answer pair; pass adapter_args.question_col/answer_col."
            )

        records: list[QuestionRecord] = []
        for idx, row in enumerate(ds):
            if min_level is not None and int(row.get("level") or 0) < int(min_level):
                continue
            gold = str(row[answer_col]).strip()
            records.append(
                QuestionRecord(
                    # "bench-" namespace guarantees benchmark qids can never
                    # collide with training/holdout qids (contamination guard).
                    qid=f"bench-{bench_name}-{idx:04d}",
                    question=str(row[question_col]).strip(),
                    # answer col may be a worked solution (math500): keep last boxed.
                    final_answer=_extract_final_answer(gold),
                    domain="math",
                    meta={"hf_name": hf_name, "row_idx": idx},
                )
            )
            if max_items and len(records) >= max_items:
                break
        print(f"[data] hf_benchmark {bench_name}: {len(records)} questions from {hf_name}[{split}]")
        return records


def load_benchmark_questions(bench) -> list[QuestionRecord]:
    """Resolve a BenchmarkCfg: explicit adapter wins, else name is a preset key.
    adapter_args always override/extend the preset."""
    if bench.adapter:
        adapter, args = bench.adapter, dict(bench.adapter_args)
    else:
        preset = BENCHMARK_PRESETS.get(bench.name)
        if preset is None:
            raise KeyError(
                f"benchmark {bench.name!r} is not a preset ({sorted(BENCHMARK_PRESETS)}) "
                "and sets no adapter; set eval.benchmarks[].adapter + adapter_args."
            )
        adapter, args = "hf_benchmark", {**preset, **bench.adapter_args}
    args.setdefault("bench_name", bench.name)
    records = build(ADAPTERS, adapter).load(args)
    # Enforce the qid namespace for ANY adapter (e.g. a local_jsonl benchmark):
    # benchmark qids must never be confusable with training/holdout qids.
    for r in records:
        if not r.qid.startswith("bench-"):
            r.qid = f"bench-{bench.name}-{r.qid}"
    return records


# ---------------------------------------------------------------------------

def load_questions(adapter_name: str, adapter_args: dict) -> list[QuestionRecord]:
    return build(ADAPTERS, adapter_name).load(adapter_args)


def ensure_questions(cfg, run_dir) -> tuple[list[QuestionRecord], list[QuestionRecord]]:
    """Materialize the train/holdout question split ONCE per run at
    runs/<name>/questions/{train,holdout}.jsonl; later stages and iterations
    read the frozen files so the split can never drift."""
    from pathlib import Path

    from .utils import is_done, mark_done

    qdir = Path(run_dir) / "questions"
    train_path, holdout_path = qdir / "train.jsonl", qdir / "holdout.jsonl"
    if not (is_done(train_path) and is_done(holdout_path)):
        records = load_questions(cfg.data.adapter, cfg.data.adapter_args)
        if not records:
            raise RuntimeError(f"adapter {cfg.data.adapter!r} produced no questions")
        train, holdout = split_holdout(records, cfg.data.eval_holdout, cfg.run.seed)
        n = QuestionRecord.dump_jsonl(train_path, train)
        mark_done(train_path, count=n, config_hash=cfg.hash())
        n = QuestionRecord.dump_jsonl(holdout_path, holdout)
        mark_done(holdout_path, count=n, config_hash=cfg.hash())
    return (
        list(QuestionRecord.load_jsonl(train_path)),
        list(QuestionRecord.load_jsonl(holdout_path)),
    )


def split_holdout(
    records: list[QuestionRecord], n_holdout: int, seed: int
) -> tuple[list[QuestionRecord], list[QuestionRecord]]:
    """Deterministic (seed, qid)-hash split, stable across iterations and
    independent of record order. Returns (train, holdout)."""
    if n_holdout <= 0:
        return list(records), []
    ranked = sorted(records, key=lambda r: stable_hash(seed, r.qid))
    holdout = ranked[:n_holdout]
    holdout_ids = {r.qid for r in holdout}
    train = [r for r in records if r.qid not in holdout_ids]
    return train, holdout


def _first_present(columns: list[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in columns:
            return c
    return None


def _extract_final_answer(gold: str) -> str:
    r"""Ground truth may be a full worked solution; keep the last \boxed{...}
    if present, else the raw string (short answers)."""
    boxed = last_boxed(gold)
    return boxed if boxed is not None else gold


def _dedup_by_qid(records: list[QuestionRecord]) -> list[QuestionRecord]:
    seen: set[str] = set()
    out = []
    for r in records:
        if r.qid not in seen:
            seen.add(r.qid)
            out.append(r)
    return out

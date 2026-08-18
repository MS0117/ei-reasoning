"""Dataset adapters -> canonical QuestionRecord list, plus the holdout split.

Adapters own all dataset-specific quirks (column names, answer normalization,
domain filtering). Downstream stages only ever see QuestionRecord.
"""

from __future__ import annotations

import re
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


# Ready-made hf_math adapter_args for the training datasets we use, selected
# via adapter_args.preset (explicit args override preset values, same merge
# rule as BENCHMARK_PRESETS). Filters differ deliberately:
#   openr1      — rich upstream metadata: drop MCQ/proof via question_type,
#                 keep only golds math-verify demonstrably graded R1
#                 generations correct against (correctness_math_verify).
#   openthoughts — gold is the last \boxed of the worked `solution`
#                 (require_boxed_gold), kept only when R1's answer was graded
#                 correct against it by upstream math-verify (`correct` col,
#                 56,730/89,120 — the same demonstrated-gradable standard as
#                 openr1; the cost is dropping problems R1 failed, but an
#                 ungradable gold is dead weight for EI either way). Proof-
#                 worded problems are excluded by regex: NuminaMath boxes the
#                 claimed bound/identity as "gold" (measured 18% of the
#                 correct=True rows!), so a model that merely echoes the
#                 statement's own expression grades as a free pass.
#   deepmath    — has none of the upstream verification metadata the other two
#                 lean on (no question_type/correct/source column), so the
#                 cleanup is expressed on the answer side: 21,725 of 103,022
#                 rows (21%) answer Yes/No/True/False or a bare MCQ letter.
#                 math-verify grades those fine, but they are guessable, so at
#                 K rollouts they reach c=0 with probability 0.5**K — never a
#                 real cliff, only noise in the pass-rate histogram. The letter
#                 half of the pattern is UPPERCASE-ONLY on purpose: [a-e] would
#                 also drop 666 legitimate algebraic golds, 406 of them "e".
#                 Proof-worded rows are NOT excluded (unlike openthoughts):
#                 final_answer is a separately curated column here rather than
#                 the last \boxed of a solution, so the echo-the-statement free
#                 pass does not apply (spot-checked: golds like \mathbb{Z}_{42},
#                 e^{t^2/2}, \ln 2 are genuine). Gold y* is r1_solution_1 with
#                 the R1 thinking half stripped (solution_strip_think), which
#                 makes it the analogue of openr1's short human `solution`:
#                 1,450 vs 14,367 chars, 100% ending in \boxed. The other two
#                 R1 solutions are genuinely different derivations (mean text
#                 similarity 0.29, 0/805 near-duplicates) and are kept in
#                 meta.gold_solutions_alt for a future multi-reference operator.
#                 difficulty/topic ride into meta (and so into the cliff set) as
#                 the analogue of the problem_type the openr1 cliff set carried.
HF_MATH_PRESETS: dict[str, dict] = {
    "openr1": {
        "hf_name": "open-r1/OpenR1-Math-220k",
        "config": "default",
        "split": "train",
        "where": {"question_type": ["math-word-problem"]},
        "require_any_true": ["correctness_math_verify"],
    },
    "openthoughts": {
        "hf_name": "open-r1/OpenThoughts-114k-math",
        "config": "default",
        "split": "train",
        "require_boxed_gold": True,
        "require_any_true": ["correct"],
        "exclude_question_regex": r"(?i)^\s*(prove|show)\b|\b(prove that|show that)\b",
    },
    "deepmath": {
        "hf_name": "zwhe99/DeepMath-103K",
        "config": "default",
        "split": "train",
        "solution_col": "r1_solution_1",
        "solution_alt_cols": ["r1_solution_2", "r1_solution_3"],
        "solution_strip_think": True,
        "meta_cols": ["difficulty", "topic"],
        # (?i:...) scopes case-folding to the words only — [A-E] must stay
        # case-SENSITIVE or it eats 666 legitimate algebraic golds (406 "e").
        "exclude_answer_regex": r"^(?:\\text\{)?(?:(?i:yes|no|true|false)|[A-E])\}?$",
    },
}


def resolve_adapter_args(adapter_name: str, args: dict) -> dict:
    """Adapter args in canonical expanded form (hf_math preset resolved,
    explicit keys winning). Cache keys over question sets must hash THIS, not
    the raw args: a preset NAME stays constant when its definition in code
    changes, so raw-args keys would keep stale frozen artifacts alive."""
    if adapter_name == "hf_math" and args.get("preset") is not None:
        base = HF_MATH_PRESETS.get(args["preset"])
        if base is None:
            raise KeyError(
                f"hf_math preset {args['preset']!r} unknown; presets: {sorted(HF_MATH_PRESETS)}"
            )
        return {**base, **{k: v for k, v in args.items() if k != "preset"}}
    return dict(args)


@register(ADAPTERS, "hf_math")
class HFMathAdapter(DatasetAdapter):
    """Generic HF math *training-data* adapter with seeded random sampling.

    Same filtering semantics as openthoughts_math (drop rows with no gold or
    a gold that math-verify cannot parse — unverifiable => useless for EI),
    but works for any HF dataset (passes `config` through to load_dataset,
    which hf_benchmark does not) and can draw a seeded pseudo-random subset.

    Answer-column preference puts `answer` BEFORE `solution`: datasets like
    open-r1/OpenR1-Math-220k carry both a clean final answer and a worked R1
    trace, and picking `solution` would leak the distillation trace into gold
    extraction.

    include_solution=True additionally copies the reference worked solution
    (column `solution_col`, default "solution") into meta.gold_solution. This
    deliberately relaxes the never-read-solutions rule BEHIND AN OPT-IN for the
    privileged improvement machinery (anchor privileged_divergence, improve
    lora_sft): y* flows only into privileged scoring contexts and transient
    LoRA weights — never into rollout/eval prompts or training text.
    solution_strip_think keeps only the worked solution half of an R1-style
    trace (see _clean_solution). solution_alt_cols names EXTRA solution columns
    cleaned the same way into meta.gold_solutions_alt (a list) — datasets like
    DeepMath ship several independent correct solutions per problem. Nothing
    consumes gold_solutions_alt yet: load_gold_solutions returns one string per
    qid, and every operator assumes that. It is captured here so a future
    multi-reference operator does not require re-deriving the question set.

    args: preset=None (HF_MATH_PRESETS key; explicit args override), hf_name
          (required unless preset), config=None, split="train",
          question_col=None, answer_col=None, n_questions=None (seeded random
          subset), seed=17, max_items=None, where=None, require_any_true=None,
          require_boxed_gold=False, include_solution=False, solution_col=None,
          solution_alt_cols=None, solution_strip_think=False, meta_cols=None.
    `max_items` head-truncates the stream BEFORE sampling (debug fast-path
    only — it biases the sample toward the head of the dataset).

    Metadata row filters (all applied BEFORE dedup/sampling; referencing a
    missing column is a hard KeyError, mirroring the config unknown-key policy):
      where: {column: [allowed values]} — keep rows whose str(column) is in the
          whitelist. E.g. {question_type: [math-word-problem]} drops OpenR1's
          MCQ rows (gold is a choice letter — letter/value mismatch misgrades,
          and 4-way guessing pollutes pass rates).
      require_any_true: [column, ...] — keep rows whose bool-list column has at
          least one true (scalar bools allowed). E.g. [correctness_math_verify]
          keeps only rows where upstream math-verify actually graded an R1
          generation correct against this gold — evidence the gold is
          verifiable in practice, which gold_parsable cannot establish (it
          accepts proof/multi-part/text golds that then misgrade).
      require_boxed_gold: drop rows whose gold has no extractable \\boxed —
          for datasets where gold is a worked solution (OpenThoughts): a
          proof-style solution without \\boxed would otherwise become a
          whole-solution-text gold that can never grade correctly.
      exclude_question_regex: drop rows whose QUESTION matches (re.search).
          For proof-worded problems ("Prove that X >= f(n)") whose "gold" is
          the statement's own expression — echoing the question grades as a
          free pass, so pass/fail carries no signal.
      exclude_answer_regex: drop rows whose EXTRACTED gold matches (re.search,
          so the pattern must carry its own ^...$ anchors). Applied to the
          post-_extract_final_answer string, i.e. exactly what grading uses.
          For coin-flip golds: DeepMath answers 21% Yes/No/True/False plus bare
          MCQ letters, which math-verify grades perfectly well but which a model
          reaches by guessing — they can never be true cliffs (0.5**K) and only
          inflate the solved class. Same rationale as openr1's question_type
          MCQ drop, expressed on the answer side because DeepMath has no
          question_type column.

    meta_cols: [column, ...] — copy these columns verbatim into meta (NOT a
        filter, but validated the same way: a missing column is a hard KeyError).
        For curation columns worth carrying into the cliff set, e.g. deepmath's
        difficulty/topic, so "which topics and difficulties does the model fall
        off?" is answerable from the cliff set alone. Values must be
        JSON-serializable — they are written to questions.jsonl.
    """

    def load(self, args: dict) -> list[QuestionRecord]:
        import datasets as hf_datasets

        from .verifier import MathVerifier

        args = resolve_adapter_args("hf_math", args)
        hf_name = args["hf_name"]
        config = args.get("config")
        split = args.get("split", "train")
        n_questions = args.get("n_questions")
        seed = args.get("seed", 17)
        max_items = args.get("max_items")
        where = {
            col: {str(v) for v in (vals if isinstance(vals, (list, tuple)) else [vals])}
            for col, vals in (args.get("where") or {}).items()
        }
        require_any_true = args.get("require_any_true") or []
        require_boxed_gold = bool(args.get("require_boxed_gold"))
        include_solution = bool(args.get("include_solution"))
        solution_col = args.get("solution_col") or "solution"
        solution_alt_cols = list(args.get("solution_alt_cols") or [])
        strip_think = bool(args.get("solution_strip_think"))
        meta_cols = list(args.get("meta_cols") or [])
        exclude_question = (
            re.compile(args["exclude_question_regex"])
            if args.get("exclude_question_regex") else None
        )
        exclude_answer = (
            re.compile(args["exclude_answer_regex"])
            if args.get("exclude_answer_regex") else None
        )

        if config:
            ds = hf_datasets.load_dataset(hf_name, config, split=split)
        else:
            ds = hf_datasets.load_dataset(hf_name, split=split)

        question_col = args.get("question_col") or _first_present(
            ds.column_names, ["problem", "question", "prompt"]
        )
        answer_col = args.get("answer_col") or _first_present(
            ds.column_names, ["answer", "final_answer", "ground_truth", "solution"]
        )
        if question_col is None or answer_col is None:
            raise KeyError(
                f"{hf_name}[{split}] columns {ds.column_names} lack a recognizable "
                "question/answer pair; pass adapter_args.question_col/answer_col."
            )
        for col in [*where, *require_any_true, *meta_cols]:
            if col not in ds.column_names:
                raise KeyError(
                    f"{hf_name}[{split}] has no column {col!r} (referenced by a "
                    f"where/require_any_true/meta_cols entry); columns: {ds.column_names}"
                )
        if include_solution:
            for col in [solution_col, *solution_alt_cols]:
                if col not in ds.column_names:
                    raise KeyError(
                        f"{hf_name}[{split}] has no column {col!r} "
                        f"(adapter_args.include_solution); columns: {ds.column_names}"
                    )

        verifier = MathVerifier()
        records: list[QuestionRecord] = []
        n_seen = n_where = n_no_true = n_q_excl = n_unboxed = n_no_gold = n_unparsable = 0
        n_a_excl = n_no_solution = n_think_stripped = 0
        for row_idx, row in enumerate(ds):
            n_seen += 1
            if any(str(row.get(col)) not in allowed for col, allowed in where.items()):
                n_where += 1
                continue
            if any(not _any_true(row.get(col)) for col in require_any_true):
                n_no_true += 1
                continue
            question = str(row.get(question_col) or "").strip()
            gold = str(row.get(answer_col) or "").strip()
            if not question or not gold:
                n_no_gold += 1
                continue
            if exclude_question is not None and exclude_question.search(question):
                n_q_excl += 1
                continue
            if require_boxed_gold and last_boxed(gold) is None:
                n_unboxed += 1
                continue
            final_answer = _extract_final_answer(gold)
            if exclude_answer is not None and exclude_answer.search(final_answer):
                n_a_excl += 1
                continue
            if not verifier.gold_parsable(final_answer):
                n_unparsable += 1
                continue
            # row_idx is the ORIGINAL dataset index (survives dedup/sampling), so
            # scripts/backfill_gold_solutions.py can positionally re-join later.
            # NB that script has no strip-think step: re-joining a solution_strip_think
            # dataset through it would write the raw R1 trace back over a clean y*.
            meta = {"hf_name": hf_name, "row_source": str(row.get("source") or ""),
                    "row_idx": row_idx}
            # Passthrough curation columns (deepmath: difficulty/topic). These
            # ride into the cliff set, which is what makes "which topics and
            # difficulties does the model fall off?" answerable without a re-join.
            for col in meta_cols:
                meta[col] = row.get(col)
            if include_solution:
                solution, stripped = _clean_solution(row.get(solution_col), strip_think)
                n_think_stripped += stripped
                if solution:
                    meta["gold_solution"] = solution
                else:
                    n_no_solution += 1
                alts = []
                for col in solution_alt_cols:
                    alt, stripped = _clean_solution(row.get(col), strip_think)
                    n_think_stripped += stripped
                    if alt:
                        alts.append(alt)
                if alts:
                    meta["gold_solutions_alt"] = alts
            records.append(
                QuestionRecord(
                    qid="hfm-" + stable_hash(hf_name, question),
                    question=question,
                    final_answer=final_answer,
                    domain="math",
                    meta=meta,
                )
            )
            if max_items and len(records) >= max_items:
                break
        records = _dedup_by_qid(records)
        print(
            f"[data] hf_math {hf_name}: kept {len(records)} / seen {n_seen} "
            f"(where-filtered {n_where}, no-true-filtered {n_no_true}, "
            f"question-excluded {n_q_excl}, answer-excluded {n_a_excl}, "
            f"unboxed-gold {n_unboxed}, "
            f"no-gold {n_no_gold}, unparsable {n_unparsable}"
            + (f", missing-solution {n_no_solution}" if include_solution else "")
            + (f", think-stripped {n_think_stripped}" if strip_think else "")
            + ")"
        )
        if n_questions:
            if n_questions > len(records):
                print(
                    f"[data] hf_math: n_questions={n_questions} > {len(records)} "
                    "available; keeping all"
                )
            else:
                # Same order-independent (seed, qid)-hash ranking as split_holdout.
                records = sorted(records, key=lambda r: stable_hash(seed, r.qid))[:n_questions]
                print(f"[data] hf_math: sampled {len(records)} questions (seed {seed})")
        return records


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


def load_gold_solutions(run_dir) -> dict[str, str]:
    """qid -> gold reference solution y* (privileged), read from the run's
    FROZEN questions/train.jsonl — the single source of truth, so downstream
    records (UnsolvedQuestion etc.) never need to carry y*. Questions without
    meta.gold_solution are simply absent from the map; stages that require y*
    decide how to handle the gap."""
    from pathlib import Path

    path = Path(run_dir) / "questions" / "train.jsonl"
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for q in QuestionRecord.load_jsonl(path):
        solution = str(q.meta.get("gold_solution") or "").strip()
        if solution:
            out[q.qid] = solution
    return out


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


THINK_CLOSE = "</think>"


def _clean_solution(text, strip_think: bool) -> tuple[str, bool]:
    r"""(reference solution as y*, whether a think region was removed).

    With strip_think, keep only what follows the LAST </think>. R1-style traces
    (DeepMath r1_solution_*) are "<reasoning> </think> <worked solution>", and
    handing the whole thing over as y* breaks three things: </think> is a REAL
    special token (151668 for Qwen3) that templates.py would re-encode mid-USER
    turn; the trace is ~10x the worked solution, so 13.5% of them alone exceed
    lora_sft's max_pair_tokens and get silently dropped; and the privileged
    scoring pass in anchor.py has no length cap, so prompt_logprobs OOMs.
    No marker => nothing to strip (the tags come in pairs), text passes through.
    """
    s = str(text or "")
    stripped = strip_think and THINK_CLOSE in s
    if stripped:
        s = s.rsplit(THINK_CLOSE, 1)[-1]
    return s.strip(), stripped


def _extract_final_answer(gold: str) -> str:
    r"""Ground truth may be a full worked solution; keep the last \boxed{...}
    if present, else the raw string (short answers)."""
    boxed = last_boxed(gold)
    return boxed if boxed is not None else gold


def _any_true(value) -> bool:
    """require_any_true semantics: bool-list columns need >= 1 true; scalars
    count as their truthiness (an absent column is caught earlier as KeyError)."""
    if isinstance(value, (list, tuple)):
        return any(bool(v) for v in value)
    return bool(value)


def _dedup_by_qid(records: list[QuestionRecord]) -> list[QuestionRecord]:
    seen: set[str] = set()
    out = []
    for r in records:
        if r.qid not in seen:
            seen.add(r.qid)
            out.append(r)
    return out

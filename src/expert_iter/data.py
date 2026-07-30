"""Dataset adapters -> canonical QuestionRecord list, plus the holdout split.

Adapters own all dataset-specific quirks (column names, answer normalization,
domain filtering). Downstream stages only ever see QuestionRecord.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .records import QuestionRecord
from .registry import ADAPTERS, build, register
from .utils import read_jsonl, stable_hash


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
    if present (balanced-brace scan), else the raw string (short answers)."""
    idx = gold.rfind("\\boxed{")
    if idx == -1:
        return gold
    start = idx + len("\\boxed{")
    depth = 1
    for i in range(start, len(gold)):
        if gold[i] == "{":
            depth += 1
        elif gold[i] == "}":
            depth -= 1
            if depth == 0:
                return gold[start:i]
    return gold  # unbalanced braces: fall back to the raw string


def _dedup_by_qid(records: list[QuestionRecord]) -> list[QuestionRecord]:
    seen: set[str] = set()
    out = []
    for r in records:
        if r.qid not in seen:
            seen.add(r.qid)
            out.append(r)
    return out

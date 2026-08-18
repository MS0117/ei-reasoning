"""The ONE place where question text becomes token ids.

Splicing rule (load-bearing for the whole pipeline): continuation prompts and
training inputs are built by concatenating token-id lists —
    prompt_ids + anchor_ids [+ continuation_ids]
— never by re-tokenizing decoded text. Re-encoding an anchor's decoded text can
merge BPE tokens across the splice point and silently shift region boundaries.
Anchor ids are always a slice of the original rollout's response_token_ids.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RenderedPrompt:
    text: str
    token_ids: list[int]


def render_question_prompt(
    tokenizer,
    question: str,
    *,
    system_prompt: str | None = None,
    question_suffix: str = "",
    chat_template_kwargs: dict | None = None,
) -> RenderedPrompt:
    """Chat-template the question as a user turn, ready for generation.

    chat_template_kwargs (from model.chat_template_kwargs) is forwarded to
    apply_chat_template for model families whose template takes switches —
    e.g. hybrid Qwen3 (Qwen3-0.6B/4B/8B) takes enable_thinking. Qwen3-*-2507
    Instruct is non-thinking: its template injects no <think> block and takes
    no enable_thinking kwarg (verified by check_env.py), so leave it empty.
    """
    return _render_user_turn(
        tokenizer, question + question_suffix,
        system_prompt=system_prompt, chat_template_kwargs=chat_template_kwargs,
    )


# Framing for the privileged teacher-forcing context (anchor stage only). The
# gold solution appears INSIDE the user turn so the same chat template applies;
# nothing downstream ever trains on this prompt.
PRIVILEGED_PREFACE = (
    "\n\nBelow is a reference solution to this problem, provided for your information:\n\n"
)
PRIVILEGED_SUFFIX = "\n\nNow solve the problem yourself, step by step."


def render_privileged_prompt(
    tokenizer,
    question: str,
    gold_solution: str,
    *,
    system_prompt: str | None = None,
    question_suffix: str = "",
    chat_template_kwargs: dict | None = None,
) -> RenderedPrompt:
    """Privileged context prompt(x, y*) for the privileged-divergence anchor:
    the question plus the gold reference solution, chat-templated. The failed
    rollout y- is appended afterwards by pure id concatenation (never here).

    This prompt must NEVER leak into rollout/eval/benchmark prompt paths or
    into training inputs — it exists only for teacher-forced scoring."""
    content = question + question_suffix + PRIVILEGED_PREFACE + gold_solution + PRIVILEGED_SUFFIX
    return _render_user_turn(
        tokenizer, content,
        system_prompt=system_prompt, chat_template_kwargs=chat_template_kwargs,
    )


def _render_user_turn(
    tokenizer,
    content: str,
    *,
    system_prompt: str | None,
    chat_template_kwargs: dict | None,
) -> RenderedPrompt:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        **(chat_template_kwargs or {}),
    )
    # Deliberately NOT apply_chat_template(tokenize=True): in transformers 5.x
    # that returns a BatchEncoding dict (check_env.py finding), and iterating
    # it yields KEYS, not ids. Encoding the rendered text is unambiguous and
    # guarantees text and token_ids describe the same sequence.
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return RenderedPrompt(text=text, token_ids=list(token_ids))


def continuation_prompt_ids(prompt_token_ids: list[int], anchor_token_ids: list[int]) -> list[int]:
    """Prompt for continuation-from-prefix generation. Pure id concatenation."""
    return list(prompt_token_ids) + list(anchor_token_ids)


def training_input_ids(
    prompt_token_ids: list[int],
    anchor_token_ids: list[int],
    completion_token_ids: list[int],
) -> list[int]:
    """Training sequence. Same id lists the model actually generated/saw."""
    return list(prompt_token_ids) + list(anchor_token_ids) + list(completion_token_ids)


# Framing for bridge generation (bridge_sft operator, generation-time ONLY):
# the base policy sees y* and produces bridge trajectories z+ that become the
# LoRA-fit targets. Nothing rendered here is ever trained on or shown at
# inference — the fit pair conditions on the PLAIN prompt (bridge_pair_ids).
BRIDGE_PREFACE = (
    "\n\nBelow is a reference solution to this problem. It is provided for your "
    "private guidance only: never mention, quote, or imply the existence of "
    "this reference solution anywhere in your output.\n\n"
)
BRIDGE_SUFFIX_ANCHORED = (
    "\n\nYou will now be shown the beginning of an attempt at this problem. "
    "Continue the attempt seamlessly from exactly where it stops — do not "
    "restart, do not repeat or rewrite any part of it, and write only the "
    "continuation. If the attempt contains an error, discover and correct it "
    "naturally within your continuation, as if you had noticed it yourself. "
    "Reason step by step to the final answer."
)
BRIDGE_SUFFIX_PLAIN = (
    "\n\nNow write your own complete step-by-step solution to the problem, "
    "ending with the final answer. You may let the reference solution guide "
    "your reasoning, but present every step as your own work."
)


def render_bridge_prompt(
    tokenizer,
    question: str,
    gold_solution: str,
    *,
    anchored: bool,
    system_prompt: str | None = None,
    question_suffix: str = "",
    chat_template_kwargs: dict | None = None,
) -> RenderedPrompt:
    """Privileged bridge-generation prompt (bridge_sft only). anchored=True
    instructs continuation of an attempt whose token ids the CALLER appends by
    pure id concatenation (continuation_prompt_ids) — never rendered here."""
    suffix = BRIDGE_SUFFIX_ANCHORED if anchored else BRIDGE_SUFFIX_PLAIN
    content = question + question_suffix + BRIDGE_PREFACE + gold_solution + suffix
    return _render_user_turn(
        tokenizer, content,
        system_prompt=system_prompt, chat_template_kwargs=chat_template_kwargs,
    )


def bridge_pair_ids(
    prompt_token_ids: list[int],
    anchor_token_ids: list[int],
    bridge_token_ids: list[int],
    eos_token_id: int,
) -> tuple[list[int], int]:
    """(x [+ a] -> z+) LoRA-fit pair: PLAIN question prompt ids + anchor ids +
    GENERATED bridge ids + EOS — pure id concatenation (z+ ids come straight
    from generation, never re-tokenized). Returns (input_ids, prompt_len) with
    loss applying to positions >= prompt_len = len(x) + len(a)."""
    ids = (list(prompt_token_ids) + list(anchor_token_ids)
           + ensure_eos(list(bridge_token_ids), eos_token_id))
    return ids, len(prompt_token_ids) + len(anchor_token_ids)


def gold_pair_ids(
    tokenizer,
    prompt_token_ids: list[int],
    solution_text: str,
    eos_token_id: int,
) -> tuple[list[int], int]:
    """(x -> y*) training pair for the transient LoRA fit (lora_fit.py):
    question prompt ids + freshly tokenized gold solution + EOS.

    Fresh tokenization of y* is legitimate here — it is not a splice of
    generated ids; it IS the training target, and this is the one boundary
    where that text becomes ids. Returns (input_ids, prompt_len); loss applies
    to positions >= prompt_len."""
    sol_ids = tokenizer(solution_text, add_special_tokens=False)["input_ids"]
    ids = list(prompt_token_ids) + ensure_eos(list(sol_ids), eos_token_id)
    return ids, len(prompt_token_ids)


def ensure_eos(token_ids: list[int], eos_token_id: int) -> list[int]:
    """Append EOS iff not already the final id (idempotent). Whether vLLM's
    output token_ids include the stop token varies by finish reason and
    sampling params, so training completions are normalized through this."""
    if not token_ids or token_ids[-1] != eos_token_id:
        return list(token_ids) + [eos_token_id]
    return list(token_ids)

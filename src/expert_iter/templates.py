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
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question + question_suffix})
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


def ensure_eos(token_ids: list[int], eos_token_id: int) -> list[int]:
    """Append EOS iff not already the final id (idempotent). Whether vLLM's
    output token_ids include the stop token varies by finish reason and
    sampling params, so training completions are normalized through this."""
    if not token_ids or token_ids[-1] != eos_token_id:
        return list(token_ids) + [eos_token_id]
    return list(token_ids)

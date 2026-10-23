"""The small local model: reads recent feedback, proposes explanation prompts."""

import random

from common import ADAPTERS, BASE_MODEL, load_json

SYSTEM = """You write the explanation instructions that a coding agent follows when it reports its work to one specific person. You are shown the current instructions and real cases where the person did or did not understand the agent's message.

Write improved instructions that would have made the failed cases understandable without hurting the ones that worked.
- Address the agent directly ("Start with...", "Define...").
- Concrete, checkable rules. No vague advice like "be clear".
- General rules that apply to any future message, not fixes for one case: never name a case's specific terms, repos or files.
- Each bullet says something new. At most 10 bullets and 160 words.
- Output only the bullet list, nothing else."""


def tidy(text: str) -> str:
    """Drops repeated and non-bullet lines, keeps at most 10 bullets."""
    seen, keep = set(), []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith(("-", "*", "•")):
            continue
        key = line.lstrip("-*• ").lower().rstrip(".")
        if key and key not in seen:
            seen.add(key)
            keep.append("- " + line.lstrip("-*• "))
    return "\n".join(keep[:10])

MAX_PROMPT_CHARS = 1500


def build_state(current: str, examples: list[dict], seed: int, n_failed: int = 8, n_worked: int = 3) -> str:
    rng = random.Random(seed)
    failed = [e for e in examples if e["reward"] < 0.5]
    worked = [e for e in examples if e["reward"] >= 0.5]
    picks = rng.sample(failed, min(n_failed, len(failed))) + rng.sample(worked, min(n_worked, len(worked)))
    cases = []
    for i, e in enumerate(picks, 1):
        excerpt = e["answer"][:500].replace("\n", " ")
        verdict = "NOT understood" if e["reward"] < 0.5 else "understood"
        cases.append(f"Case {i} ({e['type']}, {verdict})\nAgent wrote: {excerpt}...\n{e['feedback']}")
    return (
        f"Current instructions:\n{current.strip() or '(none yet)'}\n\n"
        + "\n\n".join(cases)
        + "\n\nWrite the improved instructions."
    )


def current_adapter():
    """Path of the newest trained adapter, or None before the first training."""
    v = load_json(ADAPTERS / "current.json")
    return str(ADAPTERS / v["name"]) if v else None


def propose(state: str, k: int = 4, temperature: float = 0.9) -> list[str]:
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_logits_processors, make_sampler

    model, tok = load(BASE_MODEL, adapter_path=current_adapter())
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": state}],
        add_generation_prompt=True, enable_thinking=False, tokenize=False,
    )
    sampler = make_sampler(temp=temperature, top_p=0.95)
    processors = make_logits_processors(repetition_penalty=1.15, repetition_context_size=200)
    out, seen = [], set()
    for _ in range(k * 2):
        if len(out) == k:
            break
        text = generate(model, tok, prompt=prompt, max_tokens=400, sampler=sampler, logits_processors=processors)
        text = tidy(text.split("</think>", 1)[-1])
        if text.count("\n") >= 2 and len(text) <= MAX_PROMPT_CHARS and text not in seen:
            seen.add(text)
            out.append(text)
    return out

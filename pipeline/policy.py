"""The small local model: reads recent feedback, proposes explanation prompts."""

import collections
import random
import re

from common import ADAPTERS, BASE_MODEL, load_json

# Older training pairs were whole rewritten instruction sets; they keep this
# system prompt, so the model can tell the two tasks apart.
SYSTEM_FULL = """You write the explanation instructions that a coding agent follows when it reports its work to one specific person. You are shown the current instructions and real messages the agent wrote. Each message's key items are scored for how well the person understood them: +1 understood, 0 never referenced, -1 not understood.

Write improved instructions that would have made the -1 items understandable without hurting the +1 items.
- Address the agent directly ("Start with...", "Define...").
- Concrete, checkable rules. No vague advice like "be clear".
- General rules that apply to any future message, not fixes for one case: never name a case's specific terms, repos or files.
- Each bullet says something new. At most 10 bullets and 160 words.
- Output only the bullet list, nothing else."""

SYSTEM = """You add one rule to the explanation instructions that a coding agent follows when it reports its work to one specific person. You are shown the current instructions, the most common reasons the person missed points, and real messages the agent wrote. Each message's key items are scored for how well the person understood them: +1 understood, 0 never referenced, -1 not understood.

Write ONE new rule that would have made the -1 items understandable, aimed at the most common reason. The current instructions stay as they are; your rule is added at the end.
- Address the agent directly ("Start with...", "Define...").
- Concrete and checkable. No vague advice like "be clear".
- General: never name a case's specific terms, repos or files.
- Must say something the current instructions do not already say.
- At most 40 words. Output only the rule as one line starting with "- ", nothing else."""

# Misses that better wording could fix. "wrong_or_missing" (a fact was wrong
# or left out) is not: no explanation rule can supply a missing fact.
FIXABLE = {"undefined_term", "buried", "wrong_question", "internal_ids", "analogy_story", "too_dense",
           "ambiguous_ref"}
ASK_FULL = "\n\nWrite the improved instructions."
ASK_RULE = "\n\nWrite the one new rule."
MAX_RULE_CHARS = 300


def tidy(text: str) -> str:
    """Drops repeated and non-bullet lines, keeps at most 10 bullets."""
    seen, keep = set(), []
    for line in text.splitlines():
        line = line.strip()
        body = re.sub(r"^(?:[-*•]|\d+[.)])\s*", "", line)
        if body == line:
            continue  # not a list item
        key = body.lower().rstrip(".")
        if key and key not in seen:
            seen.add(key)
            keep.append("- " + body)
    return "\n".join(keep[:10])

MAX_PROMPT_CHARS = 1500
# The state has to fit the trainer's sequence limit (with the chosen and
# rejected prompts) on a 16 GB Mac: about 1,500 tokens.
EXCERPT_CHARS = 150
ITEMS_PER_CASE = 3
ITEM_CHARS = 120
N_FAILED, N_WORKED = 4, 2


def _items(feedback: str) -> str:
    return "\n".join(line[:ITEM_CHARS] for line in feedback.splitlines()[:ITEMS_PER_CASE])


def compact(state: str) -> str:
    """Shrinks a state to the current size limits: fewer cases, shorter excerpts, fewer items.

    Applies to states written before the limits existed, so their training
    pairs stay usable; a state build_state wrote now comes back unchanged.
    """
    ask = ASK_RULE if state.endswith(ASK_RULE) else ASK_FULL
    head, *blocks = re.split(r"\n\n(?=Case \d+ \()", state.removesuffix(ask))
    failed, worked = [], []
    for b in blocks:
        m = re.match(r"Case \d+ (\(.*?, mean ([-+]\d+\.\d+)\))\nAgent wrote: (.*?)\nItems:\n(.*)", b, re.S)
        if not m:
            continue
        excerpt = m.group(3).removesuffix("...")[:EXCERPT_CHARS]
        items = _items(m.group(4))
        (failed if float(m.group(2)) < 0 else worked).append((m.group(1), excerpt, items))
    picks = failed[:N_FAILED] + worked[:N_WORKED]
    cases = [f"Case {i} {tag}\nAgent wrote: {ex}...\nItems:\n{items}" for i, (tag, ex, items) in enumerate(picks, 1)]
    return head + "\n\n" + "\n\n".join(cases) + ask


def build_state(current: str, examples: list[dict], seed: int, n_failed: int = N_FAILED, n_worked: int = N_WORKED) -> str:
    rng = random.Random(seed)
    failed = [e for e in examples if e["reward"] < 0]
    worked = [e for e in examples if e["reward"] > 0]
    picks = rng.sample(failed, min(n_failed, len(failed))) + rng.sample(worked, min(n_worked, len(worked)))
    cases = []
    for i, e in enumerate(picks, 1):
        excerpt = e["answer"][:EXCERPT_CHARS].replace("\n", " ")
        items = _items(e["feedback"])
        cases.append(f"Case {i} ({e['type']}, mean {e['reward']:+.2f})\nAgent wrote: {excerpt}...\nItems:\n{items}")
    causes = collections.Counter(it.get("cause") for e in examples for it in e["items"]
                                 if it["score"] <= -0.5 and it.get("cause") in FIXABLE)
    top = ", ".join(f"{c.replace('_', ' ')} ({n})" for c, n in causes.most_common(3)) or "unknown"
    return (
        f"Current instructions:\n{current.strip() or '(none yet)'}\n\n"
        f"Most common reasons points were missed: {top}\n\n"
        + "\n\n".join(cases)
        + ASK_RULE
    )


def one_rule(text: str) -> str | None:
    """The first bullet of a reply, or None if there is no usable one."""
    rules = tidy(text.split("</think>", 1)[-1]).splitlines()
    if not rules or len(rules[0]) > MAX_RULE_CHARS or len(rules[0]) < 20:
        return None
    return rules[0]


def add_rule(current: str, rule: str) -> str:
    return current.strip() + "\n" + rule


def current_adapter():
    """Path of the newest trained adapter, or None before the first training."""
    v = load_json(ADAPTERS / "current.json")
    if not v or not (ADAPTERS / v["name"] / "adapters.safetensors").exists():
        return None
    return str(ADAPTERS / v["name"])


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
    for _ in range(k * 3):
        if len(out) == k:
            break
        rule = one_rule(generate(model, tok, prompt=prompt, max_tokens=120, sampler=sampler, logits_processors=processors))
        if rule and rule.lower() not in seen:
            seen.add(rule.lower())
            out.append(rule)
    return out

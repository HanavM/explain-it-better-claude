"""Labels each history pair: did the agent's explanation land with the user?

Resumable: results append to data/labels.jsonl one line at a time, and pairs
already labelled are skipped on the next run.
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from common import HISTORY, LABELS, append_jsonl, claude, extract_json, load_jsonl, set_status

CAUSES = [
    "undefined_term",     # term, label, symbol, variable or threshold used without saying what it is
    "buried_answer",      # the answer or key caveat is there but not up front
    "wrong_question",     # answered something other than what was asked
    "internal_ids",       # function names, flags, hashes instead of what changed for the user
    "analogy_story",      # analogies or narrative instead of a direct statement
    "too_dense",          # too much at once: walls of tables, many numbers, many options
    "ambiguous_ref",      # "it", "the model", "0.8" could mean several things
    "wrong_or_missing",   # a fact was wrong or something important was left out
]

SYSTEM = f"""You label one moment from a person's Claude Code history: the agent wrote a message, then the person replied. Decide whether the agent's message was understood.

About this person:
- Writes terse, lowercase, many typos ("ueaj" = yes, "wdym" = what do you mean, "Wb" = what about).
- Rarely says they are confused until frustrated. Confusion shows as: re-asking something the message already answered, "wait...", "so ..., right?", "what is X", "wdym", asking to simplify, "no technical jargon", "just be straightforward".
- Often goes along with things they did not understand. "ok", "yeah go ahead", "continue" mean proceed, NOT understood.
- Understanding shows as: acting on the content correctly, restating it correctly, arguing with its substance, or explicit praise.
- Dislikes analogies and storytelling.

Watch for:
- The reply may not target this message (typed while the agent worked, answers an earlier turn, or starts an unrelated task). Then reply_targets_this is false.
- A complaint about the work (bug, wrong result, product UX) is not an explanation failure, unless the message hid or misstated the problem.
- The next reply often reveals delayed confusion; use it.

Return JSON only:
{{"reply_targets_this": bool,
  "signal": "failed" | "worked" | "unclear",
  "confidence": 0.0-1.0,
  "about": "explanation" | "work" | "both" | "neither",
  "causes": subset of {CAUSES} (only when signal is failed),
  "unclear_items": ["the specific terms or parts that did not land"],
  "evidence": "short quote from the reply"}}"""


def render(p: dict) -> str:
    parts = [
        f"<user_request>\n{p['request'][:1500]}\n</user_request>",
        f"<agent_message>\n{p['answer'][:6000]}\n</agent_message>",
        f"<reply>\n{p['reply'][:1500]}\n</reply>",
    ]
    if p.get("next_reply"):
        parts.append(f"<next_reply>\n{p['next_reply'][:800]}\n</next_reply>")
    return "\n\n".join(parts)


def label_one(p: dict) -> dict:
    out = extract_json(claude(render(p), SYSTEM, model="sonnet"))
    out["causes"] = [c for c in out.get("causes", []) if c in CAUSES]
    return {"id": p["id"], "type": p["type"], "project": p["project"], **out}


def main(limit: int | None = None, workers: int = 6) -> None:
    pairs = load_jsonl(HISTORY / "pairs.jsonl")
    done = {l["id"] for l in load_jsonl(LABELS)}
    todo = [p for p in pairs if p["id"] not in done and p["type"] != "other"][:limit]
    total = len(todo)
    print(f"{len(done)} labelled, {total} to go")
    failures = 0
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(label_one, p): p for p in todo}
        for i, f in enumerate(as_completed(futures), 1):
            try:
                append_jsonl(LABELS, f.result())
            except Exception as e:
                failures += 1
                print(f"skip {futures[f]['id']}: {e}", file=sys.stderr)
            set_status("labelling", f"{i}/{total}", done=i, total=total)
    set_status("idle", f"labelled {total - failures}, failed {failures}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)

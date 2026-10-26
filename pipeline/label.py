"""Labels each history pair item by item.

Step 1 (blind to the reply): list the key items of the agent's message, the
changes it made, results, decisions it asked for, caveats.
Step 2: score how well the user understood each item, -1 to 1, from their
reply and the message after it. 0 means the item was never referenced.

Resumable: results append to data/labels.jsonl one line at a time, and pairs
already labelled are skipped on the next run.
"""

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from common import HISTORY, LABELS, append_jsonl, claude, extract_json, load_jsonl, set_status

KINDS = ["change", "result", "decision", "caveat", "concept", "next_step"]
CAUSES = [
    "undefined_term",     # term, label, symbol, variable or threshold used without saying what it is
    "buried",             # the item is there but hidden under other material
    "wrong_question",     # answered something other than what was asked
    "internal_ids",       # function names, flags, hashes instead of what changed for the user
    "analogy_story",      # analogies or narrative instead of a direct statement
    "too_dense",          # too much at once: walls of tables, many numbers, many options
    "ambiguous_ref",      # "it", "the model", "0.8" could mean several things
    "wrong_or_missing",   # the item was misstated or something it depends on was left out
]

EXTRACT_SYSTEM = f"""You read one message a coding agent wrote to a person after doing work, and list its key items: the things the person most needs to understand from it.

Item kinds: {KINDS}.
- change: something the agent changed or built.
- result: a finding, number or outcome.
- decision: something the person must decide or answer.
- caveat: a limitation or risk that changes what the person should do.
- concept: an idea the message explains (for answers to questions).
- next_step: something the person needs to do.

Rules:
- 2 to 6 items, most important first. Skip minor details.
- Each item is one short sentence that keeps the message's own key terms, names and numbers.
- Judge from the message alone.

Return JSON only: {{"items": [{{"item": "...", "kind": "change"}}, ...]}}"""

SCORE_SYSTEM = f"""You score how well a person understood each key item of a coding agent's message, from the person's reply and the message after it.

About this person:
- Writes terse, lowercase, many typos ("ueaj" = yes, "wdym" = what do you mean, "Wb" = what about).
- Rarely says they are confused until frustrated. Confusion shows as: re-asking something the message already answered, "wait...", "so ..., right?", "what is X", "wdym", asking to simplify, "no technical jargon".
- Often goes along with things they did not understand: "ok", "yeah go ahead", "continue" are NOT evidence of understanding.
- Dislikes analogies and storytelling.

Score each item from -1 to 1:
- 1: clear evidence they understood it: acted on it correctly, restated it correctly, built on it, argued with its substance.
- 0.5: some evidence: referred to it in a way that fits a correct reading.
- 0: not referenced. No evidence either way. Use this whenever the reply does not touch the item.
- -0.5: partial misunderstanding: a checking question that reveals a gap, or a partly wrong reading.
- -1: clearly not understood: asked what it means, misread it, re-asked for it, or complained about it.

Watch for:
- The reply may not target this message at all (typed while the agent worked, answers an earlier turn, starts an unrelated task). Then reply_targets_this is false and every item scores 0.
- A complaint about the work itself (a bug, a wrong result) is not a misunderstanding of the item, unless the message hid or misstated it.
- A general complaint ("so much jargon", "i cant understand") counts against every item it plausibly covers.
- The next reply often reveals delayed confusion; use it.

For each item scored below 0, give a cause from {CAUSES}.

Return JSON only:
{{"reply_targets_this": bool,
  "items": [{{"i": 0, "score": -1|-0.5|0|0.5|1, "cause": "..." or null, "evidence": "short quote from the reply, or empty"}}, ...]}}"""


def render_message(p: dict) -> str:
    return f"<user_request>\n{p['request'][:1500]}\n</user_request>\n\n<agent_message>\n{p['answer'][:6000]}\n</agent_message>"


def render_scoring(p: dict, items: list[dict]) -> str:
    listed = "\n".join(f"{i}. [{it['kind']}] {it['item']}" for i, it in enumerate(items))
    parts = [render_message(p), f"<key_items>\n{listed}\n</key_items>", f"<reply>\n{p['reply'][:1500]}\n</reply>"]
    if p.get("next_reply"):
        parts.append(f"<next_reply>\n{p['next_reply'][:800]}\n</next_reply>")
    return "\n\n".join(parts)


def _snap(x) -> float:
    try:
        v = max(-1.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return 0.0
    return round(v * 2) / 2


def label_one(p: dict) -> dict:
    found = extract_json(claude(render_message(p), EXTRACT_SYSTEM, model="sonnet"))
    items = [
        {"item": str(it["item"]).strip(), "kind": it.get("kind") if it.get("kind") in KINDS else "result"}
        for it in found.get("items", [])
        if str(it.get("item", "")).strip()
    ][:6]
    if not items:
        raise ValueError("no items extracted")

    scored = extract_json(claude(render_scoring(p, items), SCORE_SYSTEM, model="sonnet"))
    targets = bool(scored.get("reply_targets_this", True))
    by_index = {}
    for s in scored.get("items", []):
        try:
            by_index[int(float(s["i"]))] = s
        except (KeyError, TypeError, ValueError):
            continue
    if targets and set(by_index) != set(range(len(items))):
        raise ValueError(f"scored items {sorted(by_index)} for {len(items)} items")
    for i, it in enumerate(items):
        s = by_index.get(i, {}) if targets else {}
        it["score"] = _snap(s.get("score", 0)) if targets else 0.0
        it["cause"] = s.get("cause") if it["score"] < 0 and s.get("cause") in CAUSES else None
        it["evidence"] = str(s.get("evidence") or "")[:200]
    return {"id": p["id"], "type": p["type"], "project": p["project"], "schema": 2,
            "reply_targets_this": targets, "items": items}


def main(limit: int | None = None, workers: int = 6, batch: str | None = None) -> None:
    pairs = load_jsonl(HISTORY / "pairs.jsonl")
    done = {l["id"] for l in load_jsonl(LABELS) if l.get("schema") == 2}
    todo = [p for p in pairs if p["id"] not in done and p["type"] != "other"][:limit]
    total = len(todo)
    print(f"{len(done)} labelled, {total} to go")
    failures = 0
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(label_one, p): p for p in todo}
        for i, f in enumerate(as_completed(futures), 1):
            try:
                # `batch` marks which nightly run labelled it; the backfill has none.
                append_jsonl(LABELS, {**f.result(), "batch": batch} if batch else f.result())
            except Exception as e:
                failures += 1
                print(f"skip {futures[f]['id']}: {e}", file=sys.stderr)
            set_status("labelling", f"{i}/{total}", done=i, total=total)
    set_status("idle", f"labelled {total - failures}, failed {failures}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)

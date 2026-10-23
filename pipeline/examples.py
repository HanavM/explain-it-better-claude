"""Joins labelled history and popup feedback into one list of examples.

An example is one explanation the user saw, what they said or ticked about
it, and a reward in [0, 1]. Popup answers are the user's own word and win
over a label for the same message.
"""

from common import EVENTS, HISTORY, LABELS, load_json, load_jsonl, split_of


def from_history() -> list[dict]:
    pairs = {p["id"]: p for p in load_jsonl(HISTORY / "pairs.jsonl")}
    out = []
    for l in load_jsonl(LABELS):
        p = pairs.get(l["id"])
        if p is None or not l.get("reply_targets_this") or l.get("confidence", 0) < 0.6:
            continue
        if l.get("signal") not in ("failed", "worked") or l.get("about") == "work":
            continue
        failed = l["signal"] == "failed"
        note = f'User replied: "{p["reply"][:400]}"'
        if failed:
            items = ", ".join(l.get("unclear_items") or []) or "unspecified"
            note += f"\nDid not land: {items}. Causes: {', '.join(l.get('causes') or []) or 'unspecified'}."
        else:
            note += "\nThis one was understood."
        out.append({
            "id": l["id"], "source": "history", "split": split_of(p["answer"][:2000]), "type": p["type"],
            "project": p["project"], "request": p["request"], "answer": p["answer"],
            "feedback": note, "reward": 0.0 if failed else 1.0,
            "causes": l.get("causes") or [], "unclear": l.get("unclear_items") or [],
        })
    return out


def from_popups() -> list[dict]:
    out = []
    for path in sorted(EVENTS.glob("*.json")):
        e = load_json(path)
        fb = (e or {}).get("feedback")
        if not fb or not e.get("points"):
            continue
        n = len(e["points"])
        reward = 0.0 if fb.get("none") else len(fb.get("understood", [])) / n
        lines = [f"Understood: {', '.join(fb.get('understood') or []) or 'nothing'}",
                 f"Did not understand: {', '.join(fb.get('notUnderstood') or []) or 'nothing'}"]
        if fb.get("other"):
            lines.append(f'User note: "{fb["other"]}"')
        out.append({
            "id": e["id"], "source": "popup", "split": split_of(e["answer"][:2000]),
            "type": "report" if e.get("toolCalls", 0) > 0 else "concept",
            "project": e.get("cwd", ""), "request": e.get("prompt", ""), "answer": e["answer"],
            "feedback": "\n".join(lines), "reward": reward,
            "causes": [], "unclear": fb.get("notUnderstood") or [],
        })
    return out


def all_examples() -> list[dict]:
    popups = from_popups()
    rated = {e["answer"][:2000] for e in popups}
    return popups + [e for e in from_history() if e["answer"][:2000] not in rated]

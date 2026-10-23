"""Joins labelled history and popup feedback into one list of examples.

An example is one message the user saw, its key items each scored -1 to 1
for how well the user understood it (0 = never referenced), and a reward:
the mean item score. Messages where no item has evidence either way are
left out. Popup answers are the user's own word and win over a label for
the same message.
"""

from common import EVENTS, HISTORY, LABELS, load_json, load_jsonl, split_of


def feedback_text(items: list[dict], note: str = "") -> str:
    lines = []
    for it in items:
        why = f" ({it['cause'].replace('_', ' ')})" if it.get("cause") else ""
        quote = f' — "{it["evidence"]}"' if it.get("evidence") else ""
        lines.append(f"{it['score']:+.1f} [{it['kind']}] {it['item']}{why}{quote}")
    if note:
        lines.append(note)
    return "\n".join(lines)


def _example(id_, source, answer, request, type_, project, items, note=""):
    reward = sum(it["score"] for it in items) / len(items)
    return {
        "id": id_, "source": source, "split": split_of(answer[:2000]), "type": type_, "project": project,
        "request": request, "answer": answer, "items": items, "reward": reward,
        "feedback": feedback_text(items, note),
    }


def from_history() -> list[dict]:
    pairs = {p["id"]: p for p in load_jsonl(HISTORY / "pairs.jsonl")}
    out = []
    for l in load_jsonl(LABELS):
        p = pairs.get(l["id"])
        if p is None or l.get("schema") != 2 or not l.get("reply_targets_this"):
            continue
        if not any(abs(it["score"]) >= 0.5 for it in l["items"]):
            continue  # no evidence either way
        out.append(_example(l["id"], "history", p["answer"], p["request"], p["type"], p["project"], l["items"],
                            f'User replied: "{p["reply"][:300]}"'))
    return out


def from_popups() -> list[dict]:
    out = []
    for path in sorted(EVENTS.glob("*.json")):
        e = load_json(path)
        fb = (e or {}).get("feedback")
        if not fb or not e.get("points"):
            continue
        got = set() if fb.get("none") else set(fb.get("understood") or [])
        items = [{"item": p, "kind": "result", "score": 1.0 if p in got else -1.0, "cause": None, "evidence": "popup"}
                 for p in e["points"]]
        note = f'User note: "{fb["other"]}"' if fb.get("other") else ""
        out.append(_example(e["id"], "popup", e["answer"], e.get("prompt", ""),
                            "report" if e.get("toolCalls", 0) > 0 else "concept", e.get("cwd", ""), items, note))
    return out


def all_examples() -> list[dict]:
    popups = from_popups()
    rated = {e["answer"][:2000] for e in popups}
    return popups + [e for e in from_history() if e["answer"][:2000] not in rated]

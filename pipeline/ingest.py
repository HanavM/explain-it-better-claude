"""Mines (request, agent message, reply) pairs from old Claude Code transcripts.

Only replies a person typed count: task notifications, meta rows, slash
commands, interrupts and compaction summaries are dropped, and injected
wrapper text (IDE selections, browser instructions) is stripped.
"""

import hashlib
import json
from pathlib import Path

from common import HISTORY, TRANSCRIPTS, atomic_write_text, clean_user_text


def _texts(content) -> list[str]:
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        return [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
    return []


def _is_automatic(text: str) -> bool:
    t = text.lstrip()
    return (
        t.startswith(("<task-notification", "<system-reminder", "<command-name>", "<local-command", "<bash-"))
        or "Request interrupted" in t
        or t.startswith("This session is being continued from a previous conversation")
    )


def pair_id(session: str, answer: str) -> str:
    """Stable across re-ingests: the same message always gets the same id."""
    return "h-" + hashlib.sha1((session + answer[:2000]).encode()).hexdigest()[:12]


def turn_type(tools: int, chars: int) -> str:
    if tools > 0 and chars >= 500:
        return "report"
    if tools == 0 and chars >= 1500:
        return "concept"
    return "other"


def mine_session(path: Path) -> list[dict]:
    project = path.parent.name
    rows = []
    request, agent_texts, tools, saw_agent = "", [], 0, False
    for line in path.read_text(errors="replace").splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("isSidechain"):
            continue
        msg = d.get("message") or {}
        content = msg.get("content")
        if d.get("type") == "assistant":
            saw_agent = True
            if isinstance(content, list):
                for b in content:
                    if b.get("type") == "text" and b.get("text", "").strip():
                        agent_texts.append(b["text"])
                    elif b.get("type") == "tool_use":
                        tools += 1
        elif d.get("type") == "user" and not d.get("isMeta"):
            texts = [t for t in _texts(content) if t.strip()]
            if not texts:
                continue  # tool results
            raw = "\n".join(texts)
            if _is_automatic(raw):
                continue
            human = clean_user_text(raw)
            if not human:
                continue
            if saw_agent and agent_texts:
                answer = agent_texts[-1]
                rows.append({
                    "id": pair_id(path.stem, answer),
                    "project": project,
                    "session": path.stem,
                    "ts": d.get("timestamp"),
                    "request": request[:3000],
                    "answer": answer,
                    "tools": tools,
                    "type": turn_type(tools, len(answer)),
                    "reply": human[:3000],
                })
            request, agent_texts, tools, saw_agent = human, [], 0, False
    for i, r in enumerate(rows):
        nxt = rows[i + 1] if i + 1 < len(rows) else None
        r["next_reply"] = nxt["reply"][:1500] if nxt else None
    return rows


def main() -> None:
    rows = []
    for path in sorted(TRANSCRIPTS.glob("*/*.jsonl")):
        rows.extend(mine_session(path))
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    atomic_write_text(HISTORY / "pairs.jsonl", text)
    by_type = {t: sum(r["type"] == t for r in rows) for t in ("report", "concept", "other")}
    print(f"{len(rows)} pairs {by_type}")


if __name__ == "__main__":
    main()

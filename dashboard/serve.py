"""Local dashboard: python3 dashboard/serve.py, then open http://127.0.0.1:8765"""

import calendar
import collections
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from common import (ADAPTERS, DPO, EVENTS, HISTORY, LABELS, PROMPTS, ROUNDS, STATUS, atomic_write_json, load_json,
                    load_jsonl)  # noqa: E402

PAGE = Path(__file__).with_name("index.html")
PORT = 8765


APPS = {"cli": "Terminal", "claude-vscode": "VS Code", "claude-desktop": "Desktop app"}
_session_apps: dict = {}


def session_app(session: str) -> str:
    """The app a session ran in, from its transcript (for records saved before `app` was logged)."""
    if session not in _session_apps:
        _session_apps[session] = ""
        for path in (Path.home() / ".claude" / "projects").glob(f"*/{session}.jsonl"):
            with path.open() as f:
                for line in f:
                    if '"entrypoint"' in line:
                        _session_apps[session] = json.loads(line).get("entrypoint", "")
                        break
            break
    return _session_apps[session]


def popup_list(asked: list) -> list:
    out = []
    for e in sorted(asked, key=lambda e: -e["at"]):
        app = e.get("app") or session_app(e.get("session", ""))
        out.append({
            "id": e["id"], "at": e["at"] / 1000, "trigger": e["trigger"],
            "app": APPS.get(app, app or "unknown"), "cwd": e.get("cwd", ""),
            "prompt": e.get("prompt", ""), "answer": e.get("answer", ""),
            "points": e.get("points", []), "skipped": e.get("skipped", False), "feedback": e.get("feedback"),
            "excluded": e.get("excluded"),
        })
    return out


def popups() -> dict:
    events = [e for e in (load_json(p) for p in sorted(EVENTS.glob("*.json"))) if e]
    asked = [e for e in events if e.get("trigger")]
    answered = [e for e in asked if e.get("feedback") and e.get("points") and not e.get("excluded")]
    by_day = collections.defaultdict(lambda: [0, 0])
    misses = []
    for e in answered:
        fb = e["feedback"]
        day = time.strftime("%Y-%m-%d", time.localtime(e["at"] / 1000))
        by_day[day][0] += 0 if fb.get("none") else len(fb.get("understood", []))
        by_day[day][1] += len(e["points"])
        for p in fb.get("notUnderstood", []):
            misses.append({"at": e["at"] / 1000, "text": p, "note": fb.get("other", "")})
        if fb.get("other") and not fb.get("notUnderstood"):
            misses.append({"at": e["at"] / 1000, "text": fb["other"], "note": ""})
    week = [e for e in answered if e["at"] / 1000 > time.time() - 7 * 86400]
    got = sum(0 if e["feedback"].get("none") else len(e["feedback"].get("understood", [])) for e in week)
    total = sum(len(e["points"]) for e in week)
    return {
        "logged": len(events),
        "asked": len(asked),
        "answered": len(answered),
        "skipped": sum(1 for e in asked if e.get("skipped")),
        "week_rate": got / total if total else None,
        "daily": [{"day": d, "rate": a / b} for d, (a, b) in sorted(by_day.items()) if b],
        "misses": sorted(misses, key=lambda m: -m["at"])[:8],
        "list": popup_list(asked),
    }


def labels() -> dict:
    ls = [l for l in load_jsonl(LABELS) if l.get("schema") == 2]
    items = [(l["id"], it) for l in ls if l.get("reply_targets_this") for it in l["items"]]
    causes = collections.Counter(it["cause"] for _, it in items if it["score"] < 0 and it.get("cause"))
    pairs = {p["id"]: p for p in load_jsonl(HISTORY / "pairs.jsonl")}
    misses = []
    for pid, it in items:
        if it["score"] > -0.5:
            continue
        try:
            at = calendar.timegm(time.strptime(pairs[pid]["ts"][:19], "%Y-%m-%dT%H:%M:%S"))
        except (KeyError, TypeError, ValueError):
            at = 0
        misses.append({"at": at, "text": it["item"][:140], "note": it.get("evidence", "")[:140]})
    return {
        "misses": sorted(misses, key=lambda m: -m["at"])[:8],
        "labelled": len(ls),
        "missed": sum(it["score"] < 0 for _, it in items),
        "understood": sum(it["score"] > 0 for _, it in items),
        "unreferenced": sum(it["score"] == 0 for _, it in items),
        "causes": [{"name": k.replace("_", " "), "n": v} for k, v in causes.most_common()],
    }


def rounds() -> list:
    out = []
    for p in sorted(ROUNDS.glob("r[0-9][0-9][0-9].json")):
        r = load_json(p)
        if not r:
            continue
        small = [c["val"] for c in r["candidates"] if c.get("source", "small") == "small"]
        claude = [c["val"] for c in r["candidates"] if c.get("source") == "claude"]
        out.append({
            "round": r["round"], "live": r["live"]["val"],
            "small": max(small) if small else None, "claude": claude[0] if claude else None,
            "deployed": r.get("deployed"), "source": r.get("deployed_source"),
            "pairs": r.get("new_pairs", 0), "trained": bool((r.get("training") or {}).get("ok")),
        })
    return out


def prompt() -> dict:
    cur = load_json(PROMPTS / "current.json", {"name": "v000"})
    path = PROMPTS / f"{cur['name']}.md"
    return {"name": cur["name"], "text": path.read_text().strip() if path.exists() else "", "at": cur.get("at")}


def auto() -> dict:
    import auto as sup
    state = load_json(sup.STATE, {})
    return {"new": sup.new_feedback(sup.counts(), state), "need": sup.MIN_NEW,
            "last": state.get("at"), "failed": (state.get("failed") or {}).get("at")}


def payload() -> dict:
    return {
        "auto": auto(),
        "status": load_json(STATUS, {"phase": "idle", "detail": ""}),
        "popups": popups(),
        "labels": labels(),
        "rounds": rounds(),
        "prompt": prompt(),
        "pairs": len(load_jsonl(DPO / "pairs.jsonl")),
        "adapter": load_json(ADAPTERS / "current.json"),
        "now": time.time(),
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api"):
            body, kind = json.dumps(payload()).encode(), "application/json"
        elif self.path in ("/", "/index.html"):
            body, kind = PAGE.read_bytes(), "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print(f"http://127.0.0.1:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()

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


def popups() -> dict:
    events = [e for e in (load_json(p) for p in sorted(EVENTS.glob("*.json"))) if e]
    asked = [e for e in events if e.get("trigger")]
    answered = [e for e in asked if e.get("feedback") and e.get("points")]
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
    }


def labels() -> dict:
    ls = load_jsonl(LABELS)
    usable = [l for l in ls if l.get("reply_targets_this") and l.get("confidence", 0) >= 0.6]
    causes = collections.Counter(c for l in usable if l.get("signal") == "failed" for c in l.get("causes", []))
    pairs = {p["id"]: p for p in load_jsonl(HISTORY / "pairs.jsonl")}
    misses = []
    for l in usable:
        p = pairs.get(l["id"])
        if l.get("signal") != "failed" or not p or not l.get("unclear_items"):
            continue
        try:
            at = calendar.timegm(time.strptime(p["ts"][:19], "%Y-%m-%dT%H:%M:%S"))
        except (KeyError, TypeError, ValueError):
            at = 0
        misses.append({"at": at, "text": ", ".join(l["unclear_items"])[:140], "note": l.get("evidence", "")[:140]})
    return {
        "misses": sorted(misses, key=lambda m: -m["at"])[:8],
        "labelled": len(ls),
        "failed": sum(l.get("signal") == "failed" for l in usable),
        "worked": sum(l.get("signal") == "worked" for l in usable),
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
            "proposed": r.get("proposed"), "rejected": bool(r.get("rejected")),
            "pairs": r.get("new_pairs", 0), "trained": bool((r.get("training") or {}).get("ok")),
        })
    return out


def prompt() -> dict:
    cur = load_json(PROMPTS / "current.json", {"name": "v000"})
    path = PROMPTS / f"{cur['name']}.md"
    return {"name": cur["name"], "text": path.read_text().strip() if path.exists() else "", "at": cur.get("at")}


def proposed() -> dict | None:
    p = load_json(PROMPTS / "proposed.json")
    if not p:
        return None
    path = PROMPTS / f"{p['name']}.md"
    return {**p, "text": path.read_text().strip() if path.exists() else ""}


def decide(approve: bool) -> dict:
    """Approve: the proposed prompt goes live. Reject: it is dropped."""
    p = load_json(PROMPTS / "proposed.json")
    if not p:
        return {"ok": False, "error": "nothing proposed"}
    rec_path = ROUNDS / f"r{p['round']:03d}.json"
    rec = load_json(rec_path, {})
    if approve:
        atomic_write_json(PROMPTS / "current.json", {"name": p["name"], "round": p["round"], "at": time.time()})
        rec.update({"deployed": p["name"], "deployed_source": p.get("source"), "rejected": False})
    else:
        rec["rejected"] = True
    if rec:
        atomic_write_json(rec_path, rec)
    (PROMPTS / "proposed.json").unlink(missing_ok=True)
    return {"ok": True}


def payload() -> dict:
    return {
        "status": load_json(STATUS, {"phase": "idle", "detail": ""}),
        "popups": popups(),
        "labels": labels(),
        "rounds": rounds(),
        "prompt": prompt(),
        "proposed": proposed(),
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

    def do_POST(self):
        if self.path not in ("/api/approve", "/api/reject"):
            self.send_error(404)
            return
        # Same-origin only: the page itself, never another site's form.
        origin = self.headers.get("Origin", "")
        if origin and origin != f"http://127.0.0.1:{PORT}":
            self.send_error(403)
            return
        body = json.dumps(decide(self.path == "/api/approve")).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print(f"http://127.0.0.1:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()

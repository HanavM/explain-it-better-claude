"""A/B testing explanation instructions on the user's real understanding.

No prediction: two versions take turns being live, and each is judged only on
how well the user actually understood the messages written under it.

1. Start: the current live version is A. B is A plus one new rule, written
   by Claude or by the small model (they take turns, test by test).
2. Run: once a night the live version flips to whichever has less evidence
   so far (on a tie, the other one), so both see similar days and topics.
3. Measure: each message written under a version scores the share of its key
   points the user understood, from popup answers (their own clicks) and from
   their next reply (judged by the nightly labeller).
4. Decide: resample the messages many times. Once each version has enough
   messages and one wins in at least 90% of resamples, it stays live. If no
   clear winner by the cap, A stays: a change has to earn its place.

Usage: python ab.py [status | step | start | stop]
"""

import random
import sys
import time

import policy
from common import (DATA, EVENTS, HISTORY, LABELS, PROMPTS, append_jsonl, atomic_write_json, atomic_write_text,
                    claude, load_json, load_jsonl, set_status)
from examples import all_examples

STATE = DATA / "ab.json"
HISTORY_LOG = DATA / "ab-history.jsonl"   # one line per finished test
MIN_MESSAGES = 20    # per version, before a winner can be called
MAX_MESSAGES = 60    # per version: no clear winner by then, A stays
MAX_DAYS = 28        # same, by time, for slow weeks
CONFIDENCE = 0.9     # share of resamples a winner must win
SWITCH_EVERY = 20 * 3600  # at most one flip a night (the nightly job runs ~24 h apart)


def _prompt(name: str) -> str:
    path = PROMPTS / f"{name}.md"
    return path.read_text().strip() if path.exists() else ""


def _set_live(name: str, info: dict) -> None:
    """New sessions (and new turns in open ones) pick up the switch."""
    atomic_write_json(PROMPTS / "current.json", {"name": name, "at": time.time(), **info})


def _next_name() -> str:
    used = [int(p.stem[1:]) for p in PROMPTS.glob("v[0-9][0-9][0-9].md")]
    return f"v{max(used + [0]) + 1:03d}"


def _lint_ok(rule: str) -> bool:
    import round as R  # the round module's lint check; imported late, it is heavy
    return R.lint(rule)["ok"]


def challenger(live_text: str, source: str) -> tuple[str, str] | None:
    """One new rule to add to the live version, and who wrote it."""
    examples = [e for e in all_examples() if e["split"] == "train"]
    state = policy.build_state(live_text, examples, int(time.time()))
    rules: list[str] = []
    if source == "claude":
        try:
            r = policy.one_rule(claude(state, policy.SYSTEM, model="sonnet"))
            rules = [r] if r else []
        except Exception as e:
            print(f"claude rule failed: {e}", file=sys.stderr)
    else:
        rules = policy.propose(state, k=4)
    for r in rules:
        if _lint_ok(r):
            return r, source
    return None


def start(source: str | None = None) -> dict | None:
    cur = load_json(PROMPTS / "current.json", {"name": "v000"})
    a = cur["name"]
    done = load_jsonl(HISTORY_LOG)
    if source is None:
        source = "claude" if len(done) % 2 == 0 else "small"
    set_status("ab", f"writing a challenger rule ({source})")
    got = challenger(_prompt(a), source) or challenger(_prompt(a), "claude" if source == "small" else "small")
    if not got:
        set_status("idle", "A/B: no usable challenger rule, trying again tomorrow")
        return None
    rule, source = got
    b = _next_name()
    atomic_write_text(PROMPTS / f"{b}.md", policy.add_rule(_prompt(a), rule) + "\n")
    now = time.time()
    state = {"id": len(done) + 1, "a": a, "b": b, "rule": rule, "source": source, "started": now,
             "live": b, "since": now, "status": "running"}
    atomic_write_json(STATE, state)
    _set_live(b, {"ab": state["id"]})  # B first: A already has a track record
    set_status("ab", f"test {state['id']}: {b} (A + one rule) is live")
    return state


# ---- evidence ------------------------------------------------------------

def _version_by_answer() -> dict:
    """answer prefix -> the version stamped on it by the popup plugin."""
    out = {}
    for p in EVENTS.glob("*.json"):
        e = load_json(p)
        if e and e.get("promptVersion") and e.get("answer"):
            out[e["answer"][:2000]] = e["promptVersion"]
    return out


def messages(state: dict) -> dict:
    """Each version's messages since the test started, as (time, share understood)."""
    by_answer = _version_by_answer()
    seen: set = set()
    out: dict = {state["a"]: [], state["b"]: []}

    def add(version, key, t, got, n):
        if version in out and key not in seen and n and t >= state["started"]:
            seen.add(key)
            out[version].append((t, got / n))

    # Popup answers: the user's own clicks, version stamped at the time.
    for p in EVENTS.glob("*.json"):
        e = load_json(p)
        if not e or not e.get("feedback") or not e.get("points") or e.get("excluded"):
            continue
        fb = e["feedback"]
        got = 0 if fb.get("none") else len(fb.get("understood") or [])
        add(e.get("promptVersion"), e["answer"][:2000], e["at"] / 1000, got, len(e["points"]))
    # Replies, judged by the nightly labeller. Matched to a version through the
    # popup plugin's log of the same message (every eligible one is logged).
    import datetime
    pairs = {p["id"]: p for p in load_jsonl(HISTORY / "pairs.jsonl")}
    for l in load_jsonl(LABELS):
        p = pairs.get(l["id"])
        if not p or l.get("schema") != 2 or not l.get("reply_targets_this"):
            continue
        ev = [it for it in l["items"] if abs(it["score"]) >= 0.5]
        version = by_answer.get(p["answer"][:2000])
        if not ev or not version:
            continue
        t = datetime.datetime.fromisoformat(p["ts"].replace("Z", "+00:00")).timestamp()
        add(version, p["answer"][:2000], t, sum(it["score"] > 0 for it in ev), len(ev))
    return out


def compare(a: list[float], b: list[float], seed: int = 0, n: int = 4000) -> float:
    """Share of resamples in which B's mean beats A's."""
    if not a or not b:
        return 0.5
    rng = random.Random(seed)
    wins = 0
    for _ in range(n):
        ma = sum(rng.choices(a, k=len(a))) / len(a)
        mb = sum(rng.choices(b, k=len(b))) / len(b)
        wins += mb > ma
    return wins / n


def summary(state: dict) -> dict:
    m = messages(state)
    a = [s for _, s in m[state["a"]]]
    b = [s for _, s in m[state["b"]]]
    mean = lambda x: round(sum(x) / len(x), 3) if x else None
    return {"a": {"name": state["a"], "messages": len(a), "understood": mean(a)},
            "b": {"name": state["b"], "messages": len(b), "understood": mean(b)},
            "b_better": round(compare(a, b), 3)}


# ---- the nightly step ----------------------------------------------------

def finish(state: dict, s: dict, winner: str, why: str) -> None:
    _set_live(winner, {"from_ab": state["id"]})
    state.update({"status": "done", "winner": winner, "why": why, "ended": time.time(), "result": s})
    atomic_write_json(STATE, state)
    append_jsonl(HISTORY_LOG, state)
    set_status("idle", f"A/B test {state['id']} done: {winner} stays live ({why})")


def step() -> dict | None:
    """Run once a night: start a test, flip the live version, or call a winner."""
    state = load_json(STATE)
    if not state or state.get("status") != "running":
        return start()
    s = summary(state)
    na, nb = s["a"]["messages"], s["b"]["messages"]
    days = (time.time() - state["started"]) / 86400
    if min(na, nb) >= MIN_MESSAGES and s["b_better"] >= CONFIDENCE:
        finish(state, s, state["b"], f"B better in {s['b_better']:.0%} of resamples")
    elif min(na, nb) >= MIN_MESSAGES and s["b_better"] <= 1 - CONFIDENCE:
        finish(state, s, state["a"], f"A better in {1 - s['b_better']:.0%} of resamples")
    elif min(na, nb) >= MAX_MESSAGES or days >= MAX_DAYS:
        finish(state, s, state["a"], "no clear winner, A kept")
    else:
        if time.time() - state["since"] >= SWITCH_EVERY:
            other = state["a"] if state["live"] == state["b"] else state["b"]
            # The version with less evidence goes next; a tie alternates.
            nxt = state["a"] if na < nb else state["b"] if nb < na else other
            if nxt != state["live"]:
                state.update({"live": nxt, "since": time.time()})
                _set_live(nxt, {"ab": state["id"]})
            else:
                state["since"] = time.time()
        state["last"] = s
        atomic_write_json(STATE, state)
        set_status("ab", f"test {state['id']}: {state['live']} live · A {na} msgs, B {nb} msgs")
        return state
    # A finished test starts the next one tomorrow night, so each test begins on a fresh day.
    return state


def stop() -> None:
    state = load_json(STATE)
    if state and state.get("status") == "running":
        finish(state, summary(state), state["a"], "stopped by hand, A kept")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "step":
        step()
    elif cmd == "start":
        start(sys.argv[2] if len(sys.argv) > 2 else None)
    elif cmd == "stop":
        stop()
    state = load_json(STATE)
    if state:
        import json
        print(json.dumps({**{k: state.get(k) for k in ("id", "status", "a", "b", "live", "rule", "source", "winner", "why")},
                          **({"now": summary(state)} if state.get("status") == "running" else {})}, indent=1))
    else:
        print("no A/B test yet")

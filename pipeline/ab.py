"""The RL loop, judged on real use instead of predicted scores.

One test = one RL step:

1. Propose: the small model reads recent feedback and writes K full candidate
   instruction sets. The live version joins them as the baseline.
2. Try them for real: each night one version is live for the day. Versions
   that are doing well get more days (Thompson sampling: draw a plausible
   score for each from its results so far, run the highest draw), so a clear
   loser stops wasting days early while every version still gets tried.
3. Measure: each message scores the share of its key points the user
   understood, from popup answers (their own clicks), a quick 1-5 rating
   (1 = 0, 5 = 1), and their next reply (judged by the nightly labeller). The popup plugin tags every message with
   the version that was live, so each result is credited to the right one.
4. Decide: a version wins once it is best in at least 90% of plausible draws
   and has enough messages. It goes live only if it also beats the baseline;
   otherwise the baseline stays.
5. Learn: the winning candidate becomes "chosen" and each candidate it clearly
   beat becomes "rejected" in a DPO pair, and the small model is retrained, so
   its next candidates lean toward what actually worked for the user.

Usage: python ab.py [status | step | start | stop]
"""

import json
import random
import sys
import time

import policy
from common import (DATA, DPO, EVENTS, HISTORY, LABELS, PROMPTS, append_jsonl, atomic_write_json, atomic_write_text,
                    claude, load_json, load_jsonl, set_status)
from examples import all_examples

STATE = DATA / "ab.json"
HISTORY_LOG = DATA / "ab-history.jsonl"   # one line per finished test
K = 3                # candidates per test; each one more makes a test longer
MIN_MESSAGES = 15    # the winner needs this many of its own
CONFIDENCE = 0.9     # chance of being best, for a winner
PAIR_CONFIDENCE = 0.9  # chance winner beats a candidate, for a DPO pair
PAIR_MIN = 5         # messages each side needs, for a DPO pair
MAX_DAYS = 42        # then the best so far is decided on what there is
MAX_MESSAGES = 150   # same, by total messages
SWITCH_EVERY = 20 * 3600  # at most one switch a night
DRAWS = 4000
ROUND_OFFSET = 100   # DPO pairs from test n are filed as round 100+n


def _prompt(name: str) -> str:
    path = PROMPTS / f"{name}.md"
    return path.read_text().strip() if path.exists() else ""


def _set_live(name: str, info: dict) -> None:
    """New turns pick up the switch."""
    atomic_write_json(PROMPTS / "current.json", {"name": name, "at": time.time(), **info})


def _next_name() -> str:
    used = [int(p.stem[1:]) for p in PROMPTS.glob("v[0-9][0-9][0-9].md")]
    return f"v{max(used + [0]) + 1:03d}"


# ---- starting a test -----------------------------------------------------

def start() -> dict | None:
    import round as R  # for its lint check
    cur = load_json(PROMPTS / "current.json", {"name": "v000"})
    base = cur["name"]
    done = load_jsonl(HISTORY_LOG)
    test_id = 1 + max([t["id"] for t in done] or [0])
    examples = [e for e in all_examples() if e["split"] == "train"]
    state_text = policy.build_state(_prompt(base), examples, int(time.time()))
    set_status("ab", f"test {test_id}: small model writing {K} candidates")
    texts = policy.propose(state_text, k=K + 2)  # a few spare, in case some fail the check
    set_status("ab", f"test {test_id}: checking candidates")
    arms = [{"name": base, "role": "baseline"}]
    rejected = []
    for t in texts:
        if len(arms) == K + 1:
            break
        verdict = R.lint(t)
        if not verdict["ok"]:
            rejected.append({"text": t, "problems": verdict["problems"]})
            continue
        name = _next_name()
        atomic_write_text(PROMPTS / f"{name}.md", t + "\n")
        arms.append({"name": name, "role": "candidate"})
    if len(arms) < 3:
        set_status("idle", f"A/B: only {len(arms) - 1} usable candidates, trying again tomorrow")
        return None
    now = time.time()
    state = {"id": test_id, "status": "running", "state": state_text, "baseline": base, "arms": arms,
             "failed_check": rejected, "started": now, "live": arms[1]["name"], "since": now}
    atomic_write_json(STATE, state)
    _set_live(state["live"], {"ab": test_id})
    set_status("ab", f"test {test_id}: {state['live']} live, {len(arms)} versions in the running")
    return state


# ---- evidence ------------------------------------------------------------

def messages(state: dict) -> dict:
    """Each version's messages since the test started: share of key points understood."""
    import datetime
    names = {a["name"] for a in state["arms"]}
    out: dict = {n: [] for n in names}
    seen: set = set()
    by_answer = {}
    events = [e for e in (load_json(p) for p in EVENTS.glob("*.json")) if e]
    for e in events:
        if e.get("promptVersion") and e.get("answer"):
            by_answer[e["answer"][:2000]] = e["promptVersion"]

    def add(version, key, t, got, n):
        if version in names and key not in seen and n and t >= state["started"]:
            seen.add(key)
            out[version].append(got / n)

    # Popup answers: the user's own clicks. A 1-5 rating counts as (rating-1)/4.
    for e in events:
        if e.get("kind") == "rating" and e.get("rating"):
            add(e.get("promptVersion"), e["answer"][:2000], e["at"] / 1000, e["rating"] - 1, 4)
            continue
        fb = e.get("feedback")
        if not fb or not e.get("points") or e.get("excluded"):
            continue
        got = 0 if fb.get("none") else len(fb.get("understood") or [])
        add(e.get("promptVersion"), e["answer"][:2000], e["at"] / 1000, got, len(e["points"]))
    # Replies, judged by the nightly labeller, matched to the plugin's log of the same message.
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


def _draw(scores: list[float], rng: random.Random) -> float:
    """One plausible true score, given the results so far (Beta, from a flat start)."""
    s = sum(scores)
    return rng.betavariate(1 + s, 1 + len(scores) - s)


def summary(state: dict, seed: int = 0) -> dict:
    m = messages(state)
    names = [a["name"] for a in state["arms"]]
    rng = random.Random(seed)
    best = {n: 0 for n in names}
    beats = {n: {o: 0 for o in names} for n in names}
    for _ in range(DRAWS):
        d = {n: _draw(m[n], rng) for n in names}
        best[max(d, key=d.get)] += 1
        for n in names:
            for o in names:
                beats[n][o] += d[n] > d[o]
    arms = []
    for a in state["arms"]:
        n = a["name"]
        arms.append({**a, "messages": len(m[n]), "understood": round(sum(m[n]) / len(m[n]), 3) if m[n] else None,
                     "p_best": round(best[n] / DRAWS, 3)})
    return {"arms": arms, "beats": {n: {o: round(v / DRAWS, 3) for o, v in r.items()} for n, r in beats.items()}}


# ---- the nightly step ----------------------------------------------------

def learn(state: dict, s: dict) -> dict:
    """DPO pairs from the result: the best candidate over each one it clearly beat."""
    cands = [a for a in s["arms"] if a["role"] == "candidate" and a["messages"] >= PAIR_MIN]
    if len(cands) < 2:
        return {"pairs": 0, "training": None}
    top = max(cands, key=lambda a: a["p_best"])
    n = 0
    for a in cands:
        if a is top or s["beats"][top["name"]][a["name"]] < PAIR_CONFIDENCE:
            continue
        append_jsonl(DPO / "pairs.jsonl", {"round": ROUND_OFFSET + state["id"], "source": "ab", "system": policy.SYSTEM,
                                          "prompt": state["state"], "chosen": _prompt(top["name"]),
                                          "rejected": _prompt(a["name"]),
                                          "gap": round((top["understood"] or 0) - (a["understood"] or 0), 3)})
        n += 1
    training = None
    if n:
        import round as R
        set_status("training", f"A/B test {state['id']}: learning from {n} real comparisons")
        # Only pairs decided by real use: the older predicted-score pairs are left out.
        training = R.train_policy(ROUND_OFFSET + state["id"], source="ab", min_pairs=2)
    return {"pairs": n, "training": training}


def finish(state: dict, s: dict, why: str) -> None:
    leader = max(s["arms"], key=lambda a: a["p_best"])
    base = state["baseline"]
    # A candidate goes live only if it clearly beats the baseline.
    winner = leader["name"] if leader["name"] == base or s["beats"][leader["name"]][base] >= CONFIDENCE else base
    _set_live(winner, {"from_ab": state["id"]})
    state.update({"status": "done", "winner": winner, "why": why, "ended": time.time(), "result": s})
    atomic_write_json(STATE, state)
    state["learned"] = learn(state, s)
    atomic_write_json(STATE, state)
    append_jsonl(HISTORY_LOG, {k: v for k, v in state.items() if k != "state"})
    set_status("idle", f"A/B test {state['id']} done: {winner} live ({why}); "
                       f"learned from {state['learned']['pairs']} real comparisons")


def step() -> dict | None:
    """Once a night: start a test, pick tomorrow's live version, or call a winner."""
    state = load_json(STATE)
    if not state or state.get("status") != "running":
        return start()
    s = summary(state, seed=int(time.time()))
    leader = max(s["arms"], key=lambda a: a["p_best"])
    total = sum(a["messages"] for a in s["arms"])
    days = (time.time() - state["started"]) / 86400
    if leader["p_best"] >= CONFIDENCE and leader["messages"] >= MIN_MESSAGES:
        finish(state, s, f"{leader['name']} best in {leader['p_best']:.0%} of draws")
        return state
    if days >= MAX_DAYS or total >= MAX_MESSAGES:
        finish(state, s, f"limit reached; {leader['name']} best in {leader['p_best']:.0%} of draws")
        return state
    if time.time() - state["since"] >= SWITCH_EVERY:
        rng = random.Random(int(time.time()))
        m = messages(state)
        draw = {a["name"]: _draw(m[a["name"]], rng) for a in state["arms"]}
        nxt = max(draw, key=draw.get)
        state["since"] = time.time()
        if nxt != state["live"]:
            state["live"] = nxt
            _set_live(nxt, {"ab": state["id"]})
    state["last"] = s
    atomic_write_json(STATE, state)
    set_status("ab", f"test {state['id']}: {state['live']} live · {total} messages so far")
    return state


def stop() -> None:
    """Ends the test early, decided on whatever has come in."""
    state = load_json(STATE)
    if state and state.get("status") == "running":
        finish(state, summary(state), "stopped by hand")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    {"step": step, "start": start, "stop": stop}.get(cmd, lambda: None)()
    state = load_json(STATE)
    if not state:
        print("no A/B test yet")
    else:
        show = {k: state.get(k) for k in ("id", "status", "baseline", "live", "winner", "why")}
        if state.get("status") == "running":
            show["arms"] = summary(state)["arms"]
        print(json.dumps(show, indent=1))

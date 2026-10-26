"""Supervisor: decides on its own when to train, from data alone.

Three data sources feed rounds: all past sessions (labelled once, as a
backfill), each day's sessions (labelled every night), and popup answers
(written live). launchd runs this in two ways:

- `--nightly`, once a night: mines transcripts and labels the day's new
  sessions, then checks whether a round is due.
- plain, whenever the popup writes a new record: checks whether a round is due.

A round starts once enough new feedback has built up since the last one,
whatever time it is. A popup answer is the user's own word, so it counts for
more than a label inferred from a reply.

Usage: python auto.py [--nightly] [--dry-run]
"""

import fcntl
import subprocess
import sys
import time

import ingest
import label
from common import DATA, EVENTS, LABELS, ROOT, atomic_write_json, load_json, load_jsonl, set_status

LOCK = DATA / "auto.lock"
STATE = DATA / "auto.json"       # feedback counts as of the last round, and of the last failure
MIN_NEW = 15                     # new feedback needed for a round
POPUP_WEIGHT = 3                 # one popup answer counts as this many labels


def counts() -> dict:
    labelled = sum(1 for l in load_jsonl(LABELS) if l.get("schema") == 2)
    answered = 0
    for p in EVENTS.glob("*.json"):
        e = load_json(p)
        if e and e.get("trigger") and e.get("feedback"):
            answered += 1
    return {"labelled": labelled, "answered": answered}


def new_feedback(now: dict, state: dict) -> int:
    return (now["labelled"] - state.get("labelled", 0)) + POPUP_WEIGHT * (now["answered"] - state.get("answered", 0))


def round_running() -> bool:
    return subprocess.run(["pgrep", "-f", "pipeline/round.py"], capture_output=True).returncode == 0


def main(nightly: bool = False, dry_run: bool = False) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    with open(LOCK, "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("another supervisor run is active")
            return
        if round_running():
            print("a round is already running")
            return

        if nightly and not dry_run:
            ingest.main()
            label.main(batch=time.strftime("nightly-%Y-%m-%d"))

        state = load_json(STATE, {})
        now = counts()
        new = new_feedback(now, state)
        # After a failed round, wait for fresh data rather than retrying the
        # same inputs on every wake.
        failed = state.get("failed")
        is_retry_ready = failed is None or new_feedback(now, failed) > 0
        print(f"{new}/{MIN_NEW} new feedback" + ("" if is_retry_ready else ", last round failed: waiting for new data"))
        if new < MIN_NEW or not is_retry_ready or dry_run:
            return

        started = time.time()
        r = subprocess.run([sys.executable, str(ROOT / "pipeline" / "round.py")])
        if r.returncode == 0:
            # Counts from before the round: feedback that arrived during it is new next time.
            atomic_write_json(STATE, {**now, "at": started})
        else:
            atomic_write_json(STATE, {**state, "failed": {**now, "at": time.time()}})
            set_status("failed", f"automatic round failed (exit {r.returncode}), retries when new data arrives")


if __name__ == "__main__":
    main(nightly="--nightly" in sys.argv, dry_run="--dry-run" in sys.argv)

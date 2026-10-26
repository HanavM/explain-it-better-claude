"""Hourly supervisor (run by launchd): decides on its own when to train.

Each run mines new transcripts, labels the new messages (capped, to bound
Claude calls), then starts a round once enough new feedback has built up
since the last one. A popup answer is the user's own word, so it counts
for more than a label inferred from a reply.

Usage: python auto.py [--dry-run]
"""

import fcntl
import subprocess
import sys
import time

import ingest
import label
from common import DATA, EVENTS, LABELS, ROOT, atomic_write_json, load_json, load_jsonl, set_status

LOCK = DATA / "auto.lock"
STATE = DATA / "auto.json"       # feedback counts as of the last round, and failures
MIN_NEW = 15                     # new feedback needed for a round
POPUP_WEIGHT = 3                 # one popup answer counts as this many labels
MIN_GAP_H = 6                    # at most one round per this many hours
RETRY_H = 6                      # wait after a failed round before trying again
LABEL_CAP = 60                   # labels per run


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


def main(dry_run: bool = False) -> None:
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

        if not dry_run:
            ingest.main()
            label.main(LABEL_CAP)

        state = load_json(STATE, {})
        now = counts()
        new = new_feedback(now, state)
        since_round = time.time() - state.get("at", 0)
        since_fail = time.time() - state.get("failed_at", 0)
        print(f"{new}/{MIN_NEW} new feedback, last round {since_round / 3600:.1f}h ago")
        if new < MIN_NEW or since_round < MIN_GAP_H * 3600 or since_fail < RETRY_H * 3600 or dry_run:
            return

        started = time.time()
        r = subprocess.run([sys.executable, str(ROOT / "pipeline" / "round.py")])
        if r.returncode == 0:
            # Counts from before the round: feedback that arrived during it is new next time.
            atomic_write_json(STATE, {**now, "at": started})
        else:
            atomic_write_json(STATE, {**state, "failed_at": time.time()})
            set_status("failed", f"automatic round failed (exit {r.returncode}), retrying in {RETRY_H}h")


if __name__ == "__main__":
    main("--dry-run" in sys.argv)

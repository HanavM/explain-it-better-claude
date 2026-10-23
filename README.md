# explainer-rl

RL loop that tunes how Claude Code explains its work to me.

- `explain-check/` Claude Code plugin: popup feedback + injects the live prompt
- `pipeline/` ingest → label → round (propose, replay, judge, DPO, deploy)
- `dashboard/` `python3 dashboard/serve.py` → http://127.0.0.1:8765
- `prompts/` deployed prompt versions (`current.json` points at the live one)

Run a round: `.venv/bin/python pipeline/round.py`

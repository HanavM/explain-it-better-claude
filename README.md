# explainer-rl

RL loop that tunes how Claude Code explains its work to me.

- `explain-check/` Claude Code plugin: popup feedback + injects the live prompt
- `pipeline/` ingest → label → round (propose, replay, judge, DPO, deploy)
- `dashboard/` `python3 dashboard/serve.py` → http://127.0.0.1:8765
- `prompts/` deployed prompt versions (`current.json` points at the live one)

Rounds start on their own: `pipeline/auto.py` runs hourly, mines and labels new
sessions, and starts a round once 15 new pieces of feedback have built up (a popup
answer counts as 3), at most once every 6 hours. Install it and the always-on dashboard:

```
cp launchd/*.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.hanav.explainer-rl.auto.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.hanav.explainer-rl.dashboard.plist
```

Run a round by hand: `.venv/bin/python pipeline/round.py`

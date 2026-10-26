# explainer-rl

RL loop that tunes how Claude Code explains its work to me.

- `explain-check/` Claude Code plugin: popup feedback + injects the live prompt
- `pipeline/` ingest → label → round (propose, replay, judge, DPO, deploy)
- `dashboard/` `python3 dashboard/serve.py` → http://127.0.0.1:8765
- `prompts/` deployed prompt versions (`current.json` points at the live one)

Three data sources: all past sessions (labelled once), each day's sessions (labelled
nightly at 3am, or on wake if the Mac was asleep) and popup answers. A round starts on its
own once 15 new pieces of feedback have built up since the last one (a popup answer counts
as 3); `pipeline/auto.py` checks after every nightly labelling and whenever the popup
writes a record. Install it and the always-on dashboard:

```
cp launchd/*.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.hanav.explainer-rl.auto.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.hanav.explainer-rl.nightly.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.hanav.explainer-rl.dashboard.plist
```

Run a round by hand: `.venv/bin/python pipeline/round.py`

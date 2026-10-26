"""Shared paths, file IO and the Claude call used by every pipeline step."""

import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path.home() / "explainer-rl"
DATA = ROOT / "data"
EVENTS = DATA / "events"          # written by the explain-check popup
HISTORY = DATA / "history"        # pairs mined from old transcripts
LABELS = DATA / "labels.jsonl"    # labeller output, one line per pair
ROUNDS = DATA / "rounds"          # one JSON per training round
DPO = DATA / "dpo"                # preference pairs + trainer splits
STATUS = DATA / "status.json"     # what the pipeline is doing right now
PROMPTS = ROOT / "prompts"        # v000.md, v001.md, ... and current.md
ADAPTERS = ROOT / "adapters"
TRANSCRIPTS = Path.home() / ".claude" / "projects"

BASE_MODEL = "mlx-community/Qwen3-1.7B-4bit"

# Headless calls run from an empty folder so no repo context leaks in.
_CALL_CWD = ROOT / ".cache" / "claude-cwd"


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(fd, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_jsonl(path: Path) -> list:
    """Reads every intact line; a line cut off by a crash is skipped."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def load_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def set_status(phase: str, detail: str = "", **extra) -> None:
    atomic_write_json(STATUS, {"phase": phase, "detail": detail, "at": time.time(), **extra})


def split_of(key: str) -> str:
    """Stable train/val/test split by hash, so an example never moves."""
    h = int(hashlib.sha1(key.encode()).hexdigest(), 16) % 100
    return "train" if h < 60 else "val" if h < 85 else "test"


def extract_json(text: str):
    """The first JSON object or array in a reply, fenced or not."""
    for open_, close in (("{", "}"), ("[", "]")):
        start = text.find(open_)
        end = text.rfind(close)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    raise ValueError(f"no JSON in reply: {text[:200]!r}")


def claude(prompt: str, system: str, model: str = "sonnet", timeout: int = 300, retries: int = 2) -> str:
    """One tool-less completion through the user's own Claude Code login."""
    _CALL_CWD.mkdir(parents=True, exist_ok=True)
    cmd = [
        "claude", "-p",
        "--model", model,
        "--tools", "",
        "--no-session-persistence",
        "--system-prompt", system,
    ]
    last_err = ""
    for attempt in range(retries + 1):
        try:
            r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout, cwd=_CALL_CWD,
                               env={**os.environ, "EXPLAIN_CHECK_OFF": "1"})
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()
            last_err = (r.stderr or r.stdout)[-300:]
        except subprocess.TimeoutExpired:
            last_err = "timeout"
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"claude call failed: {last_err}")


# Wrapper tags the harness injects into user turns; context, not the user's words.
_NOISE = re.compile(
    r"<(ide_opened_file|ide_selection|browser_instruction|system-reminder|pasted_content[^>]*|command-[a-z]+|local-command-[a-z]+)>.*?</\1>",
    re.S,
)
_NOISE_OPEN = re.compile(r"<(ide_opened_file|ide_selection|browser_instruction|system-reminder)>.*", re.S)


def clean_user_text(text: str) -> str:
    text = _NOISE.sub("", text)
    text = re.sub(r"<pasted_content[^>]*>.*?</pasted_content[^>]*>", "[pasted text]", text, flags=re.S)
    text = _NOISE_OPEN.sub("", text)  # a wrapper cut off by truncation
    return text.strip()

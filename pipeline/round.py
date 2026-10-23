"""One training round.

1. Propose: the small model reads feedback and writes K candidate prompts.
2. Replay: Claude rewrites held-out past messages under each candidate
   (and under the live prompt, as the baseline).
3. Judge: Claude scores each rewrite against what the user actually said.
4. Train: candidate pairs with a clear score gap become DPO pairs; the small
   model is retrained on every pair so far.
5. Deploy: the best candidate that passes a lint check goes live on its own
   only if it beats the live prompt by a margin that holds up under
   resampling, on the val split AND on the test split it never saw.

Usage: python round.py [--k 4] [--n 12] [--no-train] [--no-deploy]
"""

import argparse
import json
import random
import string
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

import policy
from common import (ADAPTERS, BASE_MODEL, DPO, PROMPTS, ROOT, ROUNDS, append_jsonl, atomic_write_json,
                    atomic_write_text, claude, extract_json, load_json, load_jsonl, set_status)
from examples import all_examples

REWRITE_SYSTEM = """You are a coding agent writing your final message to the user after finishing the work.

You get the user's request and your draft message, which holds all the facts. Rewrite the message following the EXPLANATION INSTRUCTIONS.
- Keep every fact, number, name, file path and caveat. Do not add facts or drop anything that matters.
- Change only how it is explained.
- Output only the rewritten message.

EXPLANATION INSTRUCTIONS:
{instructions}"""

JUDGE_SYSTEM = """You predict how well one specific person would understand each key item of a coding agent's message on first read.

About the person: learning as they go, often unfamiliar with statistics and internal project terms; terse; dislikes jargon, undefined labels, analogies and storytelling; wants the direct answer first.

You get: the person's request, the ORIGINAL message they saw, its KEY ITEMS with how well they understood each one in the original (+1 understood, 0 never referenced, -1 not understood, with their words as evidence), and several REWRITES of that message.

For each rewrite:
- items: for every key item in order, a score from -1 to 1 for how well this person would understand that item reading this rewrite. An item the rewrite drops, buries or misstates scores -1.
- faithful: false if it changed, added or dropped a fact, number or caveat that matters.

Judge each rewrite on its own merits. Length is not a goal in itself.
Return JSON only: {"scores": [{"id": "A", "items": [1, 0.5, -1], "faithful": true}, ...]}"""

PAIR_GAP = 1.0      # min score gap for a DPO pair
DEPLOY_MARGIN = 0.5  # min val gain over the live prompt


def live_prompt() -> tuple[str, str]:
    v = load_json(PROMPTS / "current.json", {"name": "v000"})
    path = PROMPTS / f"{v['name']}.md"
    return v["name"], path.read_text() if path.exists() else ""


def sample(examples: list[dict], split: str, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    pool = [e for e in examples if e["split"] == split]
    failed = [e for e in pool if e["reward"] < 0]
    worked = [e for e in pool if e["reward"] >= 0]
    n_failed = min(len(failed), round(n * 2 / 3))
    return rng.sample(failed, n_failed) + rng.sample(worked, min(len(worked), n - n_failed))


def rewrite(example: dict, instructions: str) -> str:
    system = REWRITE_SYSTEM.format(instructions=instructions.strip() or "(none: write it however you normally would)")
    prompt = f"<request>\n{example['request'][:2000]}\n</request>\n\n<draft>\n{example['answer'][:6000]}\n</draft>"
    return claude(prompt, system, model="sonnet")


def judge(example: dict, rewrites: list[str], seed: int) -> list[float]:
    order = list(range(len(rewrites)))
    random.Random(seed).shuffle(order)
    letters = string.ascii_uppercase
    blocks = [f"<rewrite id=\"{letters[j]}\">\n{rewrites[i][:6000]}\n</rewrite>" for j, i in enumerate(order)]
    n_items = len(example["items"])
    prompt = (
        f"<request>\n{example['request'][:1500]}\n</request>\n\n"
        f"<original>\n{example['answer'][:6000]}\n</original>\n\n"
        f"<key_items>\n{example['feedback']}\n</key_items>\n\n" + "\n\n".join(blocks)
        + f"\n\nEach rewrite's items list must have exactly {n_items} numbers, in key_items order."
    )
    got = {str(s.get("id", "")).strip().upper()[-1:]: s for s in extract_json(claude(prompt, JUDGE_SYSTEM, model="sonnet"))["scores"]}
    if set(got) != set(letters[: len(rewrites)]):
        raise ValueError(f"judge scored {sorted(got)}, expected {letters[:len(rewrites)]}")
    scores = [0.0] * len(rewrites)
    for j, i in enumerate(order):
        s = got[letters[j]]
        item_scores = [max(-1.0, min(1.0, float(x))) for x in s["items"]]
        if len(item_scores) != n_items:
            raise ValueError(f"judge gave {len(item_scores)} item scores, expected {n_items}")
        faithful = str(s.get("faithful", True)).lower() != "false"
        # 0-10: mean item understanding, mapped from [-1, 1].
        scores[i] = (sum(item_scores) / n_items + 1) * 5 * (1.0 if faithful else 0.3)
    return scores


class RoundAborted(Exception):
    pass


def evaluate(arms: list[str], batch: list[dict], seed: int, label: str) -> tuple[list[float], list[list[float]]]:
    """Mean judged score of each arm, and each judged example's per-arm scores.

    Aborts if too little of the batch was judged."""
    jobs = [(ei, ai) for ei in range(len(batch)) for ai in range(len(arms))]
    texts: dict = {}
    with ThreadPoolExecutor(6) as pool:
        futs = {pool.submit(rewrite, batch[ei], arms[ai]): (ei, ai) for ei, ai in jobs}
        for i, f in enumerate(futs, 1):
            key = futs[f]
            try:
                texts[key] = f.result()
            except Exception as e:
                print(f"rewrite failed {key}: {e}", file=sys.stderr)
            set_status("replaying", f"{label}: {i}/{len(jobs)} rewrites")
    rows: list[list[float]] = []
    with ThreadPoolExecutor(6) as pool:
        futs = {}
        for ei, ex in enumerate(batch):
            if all((ei, ai) in texts for ai in range(len(arms))):
                futs[pool.submit(judge, ex, [texts[(ei, ai)] for ai in range(len(arms))], seed + ei)] = ei
        for i, f in enumerate(futs, 1):
            try:
                rows.append(f.result())
            except Exception as e:
                print(f"judge failed: {e}", file=sys.stderr)
            set_status("judging", f"{label}: {i}/{len(futs)}")
    if len(rows) < max(2, len(batch) // 2):
        raise RoundAborted(f"{label}: only {len(rows)}/{len(batch)} examples judged")
    means = [sum(r[ai] for r in rows) / len(rows) for ai in range(len(arms))]
    return means, rows


def gain(rows: list[list[float]], a: int, b: int, seed: int, n_boot: int = 2000) -> tuple[float, float]:
    """Mean score gain of arm a over arm b, and the share of resamples where it stays above 0."""
    diffs = [r[a] - r[b] for r in rows]
    rng = random.Random(seed)
    wins = sum(sum(rng.choices(diffs, k=len(diffs))) > 0 for _ in range(n_boot))
    return sum(diffs) / len(diffs), wins / n_boot


LINT_SYSTEM = """You check explanation instructions written for a coding agent. Flag real problems only:
- names a specific project, repo, file, person or term from one past case instead of stating a general rule
- contradicts itself, or tells the agent to do something that makes explanations harder to follow (e.g. use terms without context)
- vague filler with no checkable rule ("be clear", "be precise")
- instructions about something other than how to explain work to the user
Return JSON only: {"ok": true|false, "problems": ["..."]}"""


def lint(text: str) -> dict:
    try:
        out = extract_json(claude(f"<instructions>\n{text}\n</instructions>", LINT_SYSTEM, model="sonnet"))
        return {"ok": bool(out.get("ok")), "problems": [str(p) for p in out.get("problems", [])][:5]}
    except Exception as e:
        return {"ok": False, "problems": [f"lint failed: {e}"]}


def train_policy(round_no: int) -> dict | None:
    pairs = load_jsonl(DPO / "pairs.jsonl")
    if len(pairs) < 8:
        return None
    split_dir = DPO / "split"
    rows = [{"system": p["system"], "prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]} for p in pairs]
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    held = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows[-2:])
    for name, body in (("train", text), ("valid", held), ("test", held)):
        atomic_write_text(split_dir / f"{name}.jsonl", body)
    name = f"a{round_no:03d}"
    out = ADAPTERS / name
    iters = min(150, 4 * len(rows))
    set_status("training", f"DPO on {len(rows)} pairs, {iters} steps")
    cmd = [str(ROOT / ".venv/bin/python"), "-m", "mlx_lm_lora.train",
           "--model", BASE_MODEL, "--train", "--train-mode", "dpo", "--train-type", "lora",
           "--data", str(split_dir), "--adapter-path", str(out),
           "--iters", str(iters), "--batch-size", "1", "--num-layers", "8",
           "--learning-rate", "5e-6", "--beta", "0.1", "--max-seq-length", "3072",
           "--val-batches", "1", "--steps-per-report", "10", "--save-every", str(iters),
           "--grad-checkpoint"]
    t0 = time.time()
    log = ROUNDS / f"r{round_no:03d}-train.log"
    with open(log, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    if r.returncode != 0:
        print(f"training failed, see {log}", file=sys.stderr)
        return {"ok": False, "log": str(log)}
    atomic_write_json(ADAPTERS / "current.json", {"name": name, "pairs": len(rows), "at": time.time()})
    return {"ok": True, "adapter": name, "pairs": len(rows), "iters": iters, "secs": round(time.time() - t0)}


def deploy(round_no: int, text: str, info: dict) -> str:
    """Makes the prompt live: new Claude Code sessions pick it up."""
    name = f"v{round_no:03d}"
    atomic_write_text(PROMPTS / f"{name}.md", text.strip() + "\n")
    atomic_write_json(PROMPTS / "current.json", {"name": name, "round": round_no, "at": time.time(), **info})
    return name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n", type=int, default=14)
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--no-deploy", action="store_true")
    args = ap.parse_args()

    ROUNDS.mkdir(parents=True, exist_ok=True)
    round_no = 1 + max([int(p.stem[1:4]) for p in ROUNDS.glob("r[0-9][0-9][0-9].json")] or [0])
    seed = 1000 + round_no
    started = time.time()
    record = {"round": round_no, "started": started}

    examples = all_examples()
    live_name, live_text = live_prompt()
    train_ex = [e for e in examples if e["split"] == "train"]

    set_status("proposing", f"round {round_no}: small model writing {args.k} prompts")
    state = policy.build_state(live_text, train_ex, seed)
    candidates = policy.propose(state, k=args.k)
    if len(candidates) < 2:
        set_status("idle", "too few usable candidates")
        sys.exit("too few usable candidates")
    sources = ["small"] * len(candidates)
    # Baseline: Claude writes a prompt from the same state. Its result says
    # whether the small model is earning its place, and when it wins it is a
    # good "chosen" example for the small model to learn from.
    set_status("proposing", f"round {round_no}: Claude baseline prompt")
    try:
        claude_text = policy.tidy(claude(state, policy.SYSTEM, model="sonnet"))
        if claude_text.count("\n") >= 2:
            candidates.append(claude_text)
            sources.append("claude")
    except Exception as e:
        print(f"claude baseline failed: {e}", file=sys.stderr)

    set_status("linting", f"round {round_no}: checking {len(candidates)} prompts")
    with ThreadPoolExecutor(6) as pool:
        lints = list(pool.map(lint, candidates))

    arms = [live_text] + candidates
    val = sample(examples, "val", args.n, seed)
    val_scores, val_rows = evaluate(arms, val, seed, "val")
    live_score, cand_scores = val_scores[0], val_scores[1:]
    best = max(range(len(candidates)), key=lambda i: cand_scores[i])
    eligible = [i for i in range(len(candidates)) if lints[i]["ok"]]
    pick = max(eligible, key=lambda i: cand_scores[i]) if eligible else None

    new_pairs = 0
    for i, j in combinations(range(len(candidates)), 2):
        if abs(cand_scores[i] - cand_scores[j]) >= PAIR_GAP:
            hi, lo = (i, j) if cand_scores[i] > cand_scores[j] else (j, i)
            append_jsonl(DPO / "pairs.jsonl", {"round": round_no, "system": policy.SYSTEM, "prompt": state,
                                              "chosen": candidates[hi], "rejected": candidates[lo],
                                              "gap": round(cand_scores[hi] - cand_scores[lo], 2)})
            new_pairs += 1

    record.update({
        "examples": {"total": len(examples), "popup": sum(e["source"] == "popup" for e in examples)},
        "live": {"name": live_name, "val": round(live_score, 2)},
        "candidates": [{"text": c, "val": round(s, 2), "source": src, "lint": l}
                       for c, s, src, l in zip(candidates, cand_scores, sources, lints)],
        "best": best, "new_pairs": new_pairs, "val_ids": [e["id"] for e in val],
    })

    deployed = None
    gate: dict = {"pick": pick}
    if pick is not None:
        d, p = gain(val_rows, pick + 1, 0, seed)
        gate["val"] = {"gain": round(d, 2), "p": round(p, 3)}
        if not args.no_deploy and d >= DEPLOY_MARGIN and p >= 0.9:
            test = sample(examples, "test", 12, seed)
            (t_live, t_pick), t_rows = evaluate([live_text, candidates[pick]], test, seed + 500, "test")
            td, tp = gain(t_rows, 1, 0, seed + 1)
            gate["test"] = {"live": round(t_live, 2), "pick": round(t_pick, 2), "gain": round(td, 2), "p": round(tp, 3)}
            if td > 0 and tp >= 0.8:
                deployed = deploy(round_no, candidates[pick], {"source": sources[pick], "from": live_name})
    record["gate"] = gate
    record["deployed"] = deployed
    record["deployed_source"] = sources[pick] if deployed else None

    record["training"] = None if args.no_train or new_pairs == 0 else train_policy(round_no)
    record["secs"] = round(time.time() - started)
    atomic_write_json(ROUNDS / f"r{round_no:03d}.json", record)
    set_status("idle", f"round {round_no} done" + (f", {deployed} is live" if deployed else ", live prompt kept"))
    print(json.dumps({k: record[k] for k in ("round", "live", "gate", "new_pairs", "deployed", "secs")}, indent=1))


if __name__ == "__main__":
    try:
        main()
    except RoundAborted as e:
        set_status("idle", f"round aborted: {e}")
        sys.exit(f"round aborted: {e}")
    except Exception as e:
        set_status("idle", f"round failed: {e}")
        raise

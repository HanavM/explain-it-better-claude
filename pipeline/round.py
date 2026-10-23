"""One training round.

1. Propose: the small model reads feedback and writes K candidate prompts.
2. Replay: Claude rewrites held-out past messages under each candidate
   (and under the live prompt, as the baseline).
3. Judge: Claude scores each rewrite against what the user actually said.
4. Train: candidate pairs with a clear score gap become DPO pairs; the small
   model is retrained on every pair so far.
5. Deploy: the best candidate goes live only if it beats the live prompt on
   the val split AND on the test split it never saw.

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

JUDGE_SYSTEM = """You predict whether one specific person would understand a coding agent's message on first read.

About the person: learning as they go, often unfamiliar with statistics and internal project terms; terse; dislikes jargon, undefined labels, analogies and storytelling; wants the direct answer first.

You get: the person's request, the ORIGINAL message they actually saw, what they said or ticked about it, and several REWRITES of that message. For each rewrite:
- understand: 1-10, how likely this person understands it on first read, given exactly what confused them about the original.
- faithful: false if it changed, added or dropped a fact, number or caveat that matters.

Judge each rewrite on its own merits. Length is not a goal in itself.
Return JSON only: {"scores": [{"id": "A", "understand": 7, "faithful": true}, ...]}"""

PAIR_GAP = 1.0      # min score gap for a DPO pair
DEPLOY_MARGIN = 0.5  # min val gain over the live prompt


def live_prompt() -> tuple[str, str]:
    v = load_json(PROMPTS / "current.json", {"name": "v000"})
    path = PROMPTS / f"{v['name']}.md"
    return v["name"], path.read_text() if path.exists() else ""


def sample(examples: list[dict], split: str, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    pool = [e for e in examples if e["split"] == split]
    failed = [e for e in pool if e["reward"] < 0.5]
    worked = [e for e in pool if e["reward"] >= 0.5]
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
    prompt = (
        f"<request>\n{example['request'][:1500]}\n</request>\n\n"
        f"<original>\n{example['answer'][:6000]}\n</original>\n\n"
        f"<feedback>\n{example['feedback']}\n</feedback>\n\n" + "\n\n".join(blocks)
    )
    got = {str(s.get("id", "")).strip().upper()[-1:]: s for s in extract_json(claude(prompt, JUDGE_SYSTEM, model="sonnet"))["scores"]}
    if set(got) != set(letters[: len(rewrites)]):
        raise ValueError(f"judge scored {sorted(got)}, expected {letters[:len(rewrites)]}")
    scores = [0.0] * len(rewrites)
    for j, i in enumerate(order):
        s = got[letters[j]]
        faithful = str(s.get("faithful", True)).lower() != "false"
        scores[i] = float(s["understand"]) * (1.0 if faithful else 0.3)
    return scores


class RoundAborted(Exception):
    pass


def evaluate(arms: list[str], batch: list[dict], seed: int, label: str) -> list[float]:
    """Mean judged score of each arm over the batch; aborts if too little was judged."""
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
    totals, counts = [0.0] * len(arms), [0] * len(arms)
    with ThreadPoolExecutor(6) as pool:
        futs = {}
        for ei, ex in enumerate(batch):
            if all((ei, ai) in texts for ai in range(len(arms))):
                futs[pool.submit(judge, ex, [texts[(ei, ai)] for ai in range(len(arms))], seed + ei)] = ei
        for i, f in enumerate(futs, 1):
            try:
                for ai, s in enumerate(f.result()):
                    totals[ai] += s
                    counts[ai] += 1
            except Exception as e:
                print(f"judge failed: {e}", file=sys.stderr)
            set_status("judging", f"{label}: {i}/{len(futs)}")
    if min(counts) < max(2, len(batch) // 2):
        raise RoundAborted(f"{label}: only {min(counts)}/{len(batch)} examples judged")
    return [t / c for t, c in zip(totals, counts)]


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


def deploy(round_no: int, text: str) -> str:
    name = f"v{round_no:03d}"
    atomic_write_text(PROMPTS / f"{name}.md", text.strip() + "\n")
    atomic_write_json(PROMPTS / "current.json", {"name": name, "round": round_no, "at": time.time()})
    return name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n", type=int, default=12)
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

    arms = [live_text] + candidates
    val = sample(examples, "val", args.n, seed)
    val_scores = evaluate(arms, val, seed, "val")
    live_score, cand_scores = val_scores[0], val_scores[1:]
    best = max(range(len(candidates)), key=lambda i: cand_scores[i])

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
        "candidates": [{"text": c, "val": round(s, 2), "source": src} for c, s, src in zip(candidates, cand_scores, sources)],
        "best": best, "new_pairs": new_pairs, "val_ids": [e["id"] for e in val],
    })

    deployed = None
    if not args.no_deploy and cand_scores[best] - live_score >= DEPLOY_MARGIN:
        test = sample(examples, "test", 8, seed)
        t_live, t_best = evaluate([live_text, candidates[best]], test, seed + 500, "test")
        record["test"] = {"live": round(t_live, 2), "best": round(t_best, 2)}
        if t_best > t_live:
            deployed = deploy(round_no, candidates[best])
    record["deployed"] = deployed
    record["deployed_source"] = sources[best] if deployed else None

    record["training"] = None if args.no_train or new_pairs == 0 else train_policy(round_no)
    record["secs"] = round(time.time() - started)
    atomic_write_json(ROUNDS / f"r{round_no:03d}.json", record)
    set_status("idle", f"round {round_no} done" + (f", deployed {deployed}" if deployed else ""))
    print(json.dumps({k: record[k] for k in ("round", "live", "best", "new_pairs", "deployed", "secs")}, indent=1))


if __name__ == "__main__":
    try:
        main()
    except RoundAborted as e:
        set_status("idle", f"round aborted: {e}")
        sys.exit(f"round aborted: {e}")
    except Exception as e:
        set_status("idle", f"round failed: {e}")
        raise

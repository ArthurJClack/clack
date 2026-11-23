"""teach the skill, then grade it by actually running what the model writes.

    python -m shellm tasks train --out runs/pc-skill        # ~8 min on 4 cpu cores
    python -m shellm tasks eval  runs/pc-skill/merged

grading: the model's program and the reference program are each run in a fresh
sandbox that starts from the same setup. a task passes only if both leave the
same files with the same contents and print the same output.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from .actions import Workspace, parse_program
from .suite import FAMILIES, Task, make_tasks

SPLITS = ("train", "new_names", "new_wording")


def build_data(out_dir, tokenizer, n_train: int = 24_000, n_val: int = 600, seed: int = 0) -> dict:
    """token files in the same layout the trainer reads (one <s> ... </s> per task)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def enc(tasks):
        ids = tokenizer.encode_batch([t.text() for t in tasks])
        return np.concatenate([np.array([tokenizer.bos_id, *e, tokenizer.eos_id], dtype=np.uint16) for e in ids])

    train = enc(make_tasks(n_train, seed, "train"))
    val = enc(make_tasks(n_val, seed + 1, "train"))
    train.tofile(out / "train.bin")
    val.tofile(out / "val.bin")
    meta = {"dtype": "uint16", "vocab_size": tokenizer.vocab_size, "tokenizer": str(Path(tokenizer.path).resolve()),
            "train_tokens": int(train.size), "val_tokens": int(val.size), "files": n_train, "kind": "pc-tasks"}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def _outcome(task: Task, program: list[str]) -> tuple:
    with tempfile.TemporaryDirectory() as d:
        ws = Workspace(d)
        for line in task.setup:
            ws.run(line)
        log = ws.run_program(program)
        return ws.snapshot(), [(e["ok"], e["output"]) for e in log]


def grade(task: Task, program: list[str]) -> bool:
    return bool(program) and _outcome(task, program) == _outcome(task, task.program)


def propose(session, instruction: str, max_tokens: int = 60) -> list[str]:
    """ask the model for a program. greedy, and no repetition penalty: copying names is the whole job."""
    text = session.complete(f"task: {instruction}\n>", max_tokens=max_tokens, temperature=0.0, repetition_penalty=1.0,
                            stop=["\n.\n", "\ntask:"], source="pc-task")
    return parse_program(">" + text)


def evaluate(session, n: int = 160, seed: int = 123, splits=SPLITS, verbose: bool = False) -> dict:
    res: dict = {}
    for split in splits:
        tasks = make_tasks(n, seed, split)
        per: dict[str, list[int]] = {f: [] for f in FAMILIES}
        for t in tasks:
            prog = propose(session, t.instruction)
            ok = grade(t, prog)
            per[t.family].append(int(ok))
            if verbose and not ok:
                print(f"  ✗ [{split}/{t.family}] {t.instruction!r}\n      wanted {t.program}\n      got    {prog}")
        res[split] = {"pass_rate": float(np.mean([x for v in per.values() for x in v])),
                      "by_family": {f: float(np.mean(v)) for f, v in per.items() if v}, "n": len(tasks)}
    return res


def train_skill(base: str, out: str, steps: int = 900, lora_r: int = 32, n_train: int = 24_000, lr: float = 2e-3,
                batch_size: int = 32, seq_len: int = 128, seed: int = 0) -> Path:
    from ..io import load_tokenizer
    from ..train import TrainArgs, train

    out_p = Path(out)
    build_data(out_p / "data", load_tokenizer(base), n_train, seed=seed)
    return train(TrainArgs(data=str(out_p / "data"), out=str(out_p), init_from=str(base), steps=steps, batch_size=batch_size,
                           seq_len=seq_len, lr=lr, warmup=30, weight_decay=0.0, eval_every=150, eval_batches=10,
                           lora_r=lora_r, lora_alpha=float(lora_r), seed=seed))

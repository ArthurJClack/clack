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
from .suite import FAMILIES, NAMES_HELD, Task, make_tasks

SPLITS = ("train", "new_names", "new_wording")
PC_MODEL = "shellm-nano-pc"


def load_pc_session(name: str | None = None, base: str | None = None, device: str = "auto"):
    """the model that knows the action language: a full checkpoint (default: the bundled
    shellm-nano-pc, or your own in ~/.shellm/models) or a lora adapter added to `base`."""
    from ..api import Session
    from ..io import resolve_ckpt

    name = name or PC_MODEL
    try:
        return Session(resolve_ckpt(name), device=device)
    except FileNotFoundError:
        resolve_ckpt(name, marker="adapter_config.json")  # raises FileNotFoundError if it is neither
        return Session.with_skill(name, base=base, device=device)


def install_skill(ckpt, name: str = PC_MODEL) -> Path:
    """copy a trained skill model to ~/.shellm/models so /do and `shellm do` use it."""
    import shutil

    dst = Path.home() / ".shellm" / "models" / name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(ckpt, dst)
    return dst


def vocab_words(tokenizer) -> list[str]:
    """every whole lowercase word the tokenizer knows, minus the held-out test words.
    training names are drawn from all of them: with thousands of different words to
    copy, memorizing stops working and the model has to learn to copy."""
    import random

    toks, _ = tokenizer.vocab_and_merges()
    held = set(NAMES_HELD)
    ok = lambda w: w.isascii() and w.isalpha() and w.islower()
    words = sorted({t[1:] for t in toks if t.startswith("\u0120") and len(t) >= 5 and ok(t[1:])} - held)
    starts = sorted({t[1:] for t in toks if t.startswith("\u0120") and len(t) >= 3 and ok(t[1:])})
    inner = sorted({t for t in toks if 2 <= len(t) <= 5 and ok(t)})
    # most real words are several pieces to the model ("sketches" = sk + et + ches). build
    # multi-piece names out of every piece it has, so no piece is new when it must be copied.
    rng = random.Random(0)
    glued = {rng.choice(starts) + "".join(rng.choice(inner) for _ in range(rng.randint(1, 2))) for _ in range(len(words) * 3)}
    return words + sorted(w for w in glued - held if 4 <= len(w) <= 12)


def build_data(out_dir, tokenizer, n_train: int = 40_000, n_val: int = 600, seed: int = 0) -> dict:
    """token files in the same layout the trainer reads (one <s> ... </s> per task)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pool = vocab_words(tokenizer)

    def enc(tasks):
        ids = tokenizer.encode_batch([t.text() for t in tasks])
        return np.concatenate([np.array([tokenizer.bos_id, *e, tokenizer.eos_id], dtype=np.uint16) for e in ids])

    train = enc(make_tasks(n_train, seed, "train", pool=pool))
    val = enc(make_tasks(n_val, seed + 1, "train", pool=pool))
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


GRAMMAR = "\n> mkdir write append read list count copy move delete find / . *.txt *.md *.log *.csv\n.\n"


def propose(session, instruction: str, max_tokens: int = 60, constrain: bool = True) -> list[str]:
    """ask the model for a program. greedy, and no repetition penalty: copying names is the whole job.

    constrain=True only lets it produce tokens that are in the instruction or in the
    action grammar. a model this small picks the right actions but garbles names it
    has never seen ("thunder8.txt" -> "eum8.txt"); it cannot garble what it cannot emit."""
    allowed = None
    if constrain:
        tk = session.tk
        allowed = sorted(set(tk.encode(" " + instruction)) | set(tk.encode(instruction)) | set(tk.encode(GRAMMAR)))
    text = session.complete(f"task: {instruction}\n>", max_tokens=max_tokens, temperature=0.0, repetition_penalty=1.0,
                            stop=["\n.\n", "\ntask:"], source="pc-task", allowed_ids=allowed)
    return parse_program(">" + text)


def evaluate(session, n: int = 160, seed: int = 123, splits=SPLITS, verbose: bool = False, constrain: bool = True) -> dict:
    res: dict = {}
    for split in splits:
        tasks = make_tasks(n, seed, split)
        per: dict[str, list[int]] = {f: [] for f in FAMILIES}
        for t in tasks:
            prog = propose(session, t.instruction, constrain=constrain)
            ok = grade(t, prog)
            per[t.family].append(int(ok))
            if verbose and not ok:
                print(f"  ✗ [{split}/{t.family}] {t.instruction!r}\n      wanted {t.program}\n      got    {prog}")
        res[split] = {"pass_rate": float(np.mean([x for v in per.values() for x in v])),
                      "by_family": {f: float(np.mean(v)) for f, v in per.items() if v}, "n": len(tasks)}
    return res


def train_skill(base: str, out: str, steps: int = 2600, lora_r: int = 0, n_train: int = 60_000, lr: float | None = None,
                batch_size: int = 32, seq_len: int = 128, seed: int = 0) -> Path:
    """lora_r=0 trains every weight (needed to really learn copying);
    lora_r>0 trains a small adapter instead (a few mb, but it underfits this skill)."""
    lr = lr or (2e-3 if lora_r else 6e-4)
    from ..io import load_tokenizer
    from ..train import TrainArgs, train

    out_p = Path(out)
    build_data(out_p / "data", load_tokenizer(base), n_train, seed=seed)
    return train(TrainArgs(data=str(out_p / "data"), out=str(out_p), init_from=str(base), steps=steps, batch_size=batch_size,
                           seq_len=seq_len, lr=lr, warmup=30, weight_decay=0.0, eval_every=150, eval_batches=10,
                           lora_r=lora_r, lora_alpha=float(lora_r), seed=seed))

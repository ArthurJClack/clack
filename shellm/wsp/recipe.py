"""run a weight-space program.

a recipe is a plain python file executed with the whole wsp toolkit already in
scope (load, ties, rome, steering_vector, low_rank, loss, table, ...). anything
it saves records the recipe path + source in info.json, so every model knows
how it was made.

    python -m shellm wsp recipes/01_task_vectors.py
"""
from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path

import torch

from . import edit, merge, steer, surgery
from .tensors import Weights, load

RECIPE_CONTEXT: dict = {}


def loss(W: Weights, data, batches: int = 30) -> float:
    from ..evaluate import loss_on

    return loss_on(W.model(), data, batches)


def table(rows: list[dict], title: str | None = None) -> str:
    if not rows:
        return ""
    cols = list(rows[0])
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
    w = {c: max(len(c), *(len(fmt(r.get(c, ""))) for r in rows)) for c in cols}
    lines = ([title] if title else []) + ["  ".join(c.ljust(w[c]) for c in cols), "  ".join("-" * w[c] for c in cols)]
    lines += ["  ".join(fmt(r.get(c, "")).ljust(w[c]) for c in cols) for r in rows]
    s = "\n".join(lines)
    print(s)
    return s


def namespace(args: list[str] | None = None) -> dict:
    ns = {
        "torch": torch, "math": math, "Path": Path, "args": args or [],
        "Weights": Weights, "load": load, "loss": loss, "table": table,
        "merge": merge, "edit": edit, "steer": steer, "surgery": surgery,
    }
    for mod in (merge, edit, steer, surgery):
        for k in dir(mod):
            if not k.startswith("_") and callable(getattr(mod, k)) and getattr(getattr(mod, k), "__module__", "") == mod.__name__:
                ns[k] = getattr(mod, k)
    ns["Fact"] = edit.Fact
    return ns


def run_recipe(path, args: list[str] | None = None) -> dict:
    p = Path(path)
    src = p.read_text()
    RECIPE_CONTEXT.clear()
    RECIPE_CONTEXT.update({"recipe": str(p), "recipe_sha256": hashlib.sha256(src.encode()).hexdigest()[:16], "recipe_source": src})
    ns = namespace(args)
    ns["__file__"] = str(p)
    ns["__name__"] = "__recipe__"
    t0 = time.time()
    print(f"[wsp] running {p}")
    exec(compile(src, str(p), "exec"), ns)
    print(f"[wsp] done in {time.time() - t0:.1f}s")
    RECIPE_CONTEXT.clear()
    return ns

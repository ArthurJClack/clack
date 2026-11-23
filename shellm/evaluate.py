"""loss / perplexity on token data."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from .data import TokenData


@torch.no_grad()
def loss_on(model, data, batches: int = 40, seq_len: int = 256, batch_size: int = 16) -> float:
    td = data if isinstance(data, TokenData) else TokenData(data, "val")
    model.eval()
    ls = [float(model(x, y)[1]) for x, y in td.sequential(batch_size, min(seq_len, model.cfg.max_seq_len), batches)]
    return float(np.mean(ls))


def eval_ckpt(ckpt, data_dirs: list[str], batches: int = 40, seq_len: int = 256) -> dict[str, float]:
    from .io import load_model

    model, _ = load_model(ckpt)
    out = {}
    for d in data_dirs:
        l = loss_on(model, d, batches, seq_len)
        out[f"{Path(d).name}_loss"] = l
        out[f"{Path(d).name}_ppl"] = math.exp(l)
    return out

"""structural weight surgery.

  low_rank(W, rank=32)                 svd-truncate matrices (or energy=0.9)
  prune(W, 0.5)                        magnitude pruning
  fake_quant(W, bits=4, group=32)      see what n-bit quantization does before exporting
  stack_layers(W, [0,1,2,3,2,3,4,5])   drop / duplicate / reorder blocks ("frankenmerge")
  transplant(dst, src, "*.mlp.*")      copy tensors from another model
  splice(a, b, n)                      first n blocks from a, the rest from b
"""
from __future__ import annotations

import copy
import re

import torch

from .tensors import Weights

BIG = ("*.self_attn.*_proj.weight", "*.mlp.*_proj.weight")


def low_rank(W: Weights, rank: int | None = None, energy: float | None = None, patterns=BIG) -> Weights:
    if (rank is None) == (energy is None):
        raise ValueError("give exactly one of rank= or energy=")

    def f(_, v):
        U, S, Vh = torch.linalg.svd(v.float(), full_matrices=False)
        if energy is not None:
            e = (S**2).cumsum(0) / (S**2).sum()
            r = int((e < energy).sum()) + 1
        else:
            r = min(rank, S.numel())
        return (U[:, :r] * S[:r]) @ Vh[:r]

    return W.map(f, *patterns)


def spectrum(W: Weights, name: str) -> torch.Tensor:
    return torch.linalg.svdvals(W.t[name].float())


def prune(W: Weights, sparsity: float = 0.5, patterns=BIG) -> Weights:
    def f(_, v):
        k = int(v.numel() * sparsity)
        if k == 0:
            return v
        th = v.abs().reshape(-1).kthvalue(k).values
        return torch.where(v.abs() > th, v, torch.zeros_like(v))

    return W.map(f, *patterns)


def fake_quant(W: Weights, bits: int = 4, group: int = 32, patterns=("*.weight",)) -> Weights:
    """symmetric per-group round-to-nearest, like gguf q4_0/q8_0."""
    qmax = 2 ** (bits - 1) - 1

    def f(_, v):
        if v.dim() < 2 or v.shape[-1] % group:
            return v
        g = v.reshape(-1, group)
        s = g.abs().amax(-1, keepdim=True).clamp(min=1e-12) / qmax
        return (torch.round(g / s).clamp(-qmax - 1, qmax) * s).reshape(v.shape)

    return W.map(f, *patterns)


_LAYER = re.compile(r"^model\.layers\.(\d+)\.(.*)$")


def stack_layers(W: Weights, order: list[int]) -> Weights:
    """build a model whose block i is a copy of W's block order[i]."""
    out = {k: v.clone() for k, v in W.items() if not _LAYER.match(k)}
    for new_i, old_i in enumerate(order):
        for k, v in W.items():
            m = _LAYER.match(k)
            if m and int(m.group(1)) == old_i:
                out[f"model.layers.{new_i}.{m.group(2)}"] = v.clone()
    cfg = copy.deepcopy(W.cfg)
    cfg.n_layers = len(order)
    return Weights(out, cfg, W.tokenizer, W.kind, {**W.info, "layer_order": order})


def transplant(dst: Weights, src: Weights, *patterns: str) -> Weights:
    out = dst.copy()
    for k, v in src.select(*patterns).items():
        out.t[k] = v.clone()
    return out


def splice(a: Weights, b: Weights, n: int) -> Weights:
    """blocks [0, n) from a, [n, L) from b; embeddings from a, final norm from b."""
    out = a.copy()
    for k, v in b.items():
        m = _LAYER.match(k)
        if (m and int(m.group(1)) >= n) or k == "model.norm.weight":
            out.t[k] = v.clone()
    return out

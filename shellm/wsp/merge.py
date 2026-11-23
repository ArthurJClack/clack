"""model merging in weight space.

  task_vector(ft, base)              ft - base
  linear([a, b], [0.5, 0.5])         weighted average (a "model soup")
  slerp(a, b, t)                     spherical interpolation, per tensor
  ties(base, [ft1, ft2], density)    trim -> elect sign -> disjoint merge  (yadav et al. 2023)
  dare(delta, p)                     drop p of the delta at random, rescale (yu et al. 2023)
  dare_ties(base, [ft1, ft2], p)     dare each delta, then ties-merge
"""
from __future__ import annotations

import torch

from .tensors import Weights


def task_vector(ft: Weights, base: Weights) -> Weights:
    return ft - base


def linear(models: list[Weights], weights: list[float] | None = None) -> Weights:
    weights = weights or [1.0 / len(models)] * len(models)
    out = models[0] * weights[0]
    for m, w in zip(models[1:], weights[1:]):
        out = out + m * w
    return out


def _slerp(a: torch.Tensor, b: torch.Tensor, t: float, eps: float = 1e-8) -> torch.Tensor:
    af, bf = a.float().reshape(-1), b.float().reshape(-1)
    an, bn = af / (af.norm() + eps), bf / (bf.norm() + eps)
    dot = (an * bn).sum().clamp(-1.0, 1.0)
    if dot.abs() > 0.9995:  # nearly parallel -> lerp
        return (1 - t) * a + t * b
    om = torch.acos(dot)
    so = torch.sin(om)
    return ((torch.sin((1 - t) * om) / so) * af + (torch.sin(t * om) / so) * bf).reshape(a.shape).to(a.dtype)


def slerp(a: Weights, b: Weights, t: float = 0.5) -> Weights:
    return a.like({k: _slerp(v, b.t[k], t) if v.dim() >= 2 else (1 - t) * v + t * b.t[k] for k, v in a.items()})


def _deltas(base: Weights, models: list[Weights]) -> list[Weights]:
    return [m if m.kind == "delta" else m - base for m in models]


def trim(delta: Weights, density: float) -> Weights:
    """keep the top `density` fraction of each tensor by magnitude."""
    def f(_, v):
        if density >= 1.0:
            return v
        k = max(1, int(v.numel() * density))
        thresh = v.abs().reshape(-1).float().kthvalue(v.numel() - k + 1).values
        return torch.where(v.abs() >= thresh, v, torch.zeros_like(v))

    return delta.map(f)


def ties(base: Weights, models: list[Weights], density: float = 0.2, weights: list[float] | None = None, scale: float = 1.0) -> Weights:
    deltas = [trim(d, density) for d in _deltas(base, models)]
    weights = weights or [1.0] * len(deltas)
    merged = {}
    for k in deltas[0].keys():
        stack = torch.stack([d.t[k] * w for d, w in zip(deltas, weights)])
        sign = torch.sign(stack.sum(0))  # elect: the sign with more total mass wins
        agree = (torch.sign(stack) == sign) & (stack != 0)
        num = (stack * agree).sum(0)
        den = agree.sum(0).clamp(min=1)
        merged[k] = num / den
    return base + base.like(merged, kind="delta") * scale


def dare(delta: Weights, p: float = 0.9, seed: int = 0, rescale: bool = True) -> Weights:
    g = torch.Generator().manual_seed(seed)

    def f(_, v):
        keep = (torch.rand(v.shape, generator=g) >= p).to(v.dtype)
        return v * keep / (1 - p) if rescale else v * keep

    return delta.map(f)


def dare_ties(base: Weights, models: list[Weights], p: float = 0.9, weights: list[float] | None = None, seed: int = 0) -> Weights:
    deltas = [dare(d, p, seed + i) for i, d in enumerate(_deltas(base, models))]
    return ties(base, deltas, density=1.0, weights=weights)


def interference(deltas: list[Weights]) -> dict:
    """how much do task vectors fight? pairwise cosine + sign-conflict rate."""
    out = {}
    for i in range(len(deltas)):
        for j in range(i + 1, len(deltas)):
            a, b = deltas[i], deltas[j]
            keys = sorted(set(a.keys()) & set(b.keys()))
            a, b = a.select(*keys), b.select(*keys)
            fa, fb = a.flat(keys), b.flat(keys)
            both = (fa != 0) & (fb != 0)
            conflict = float(((torch.sign(fa) != torch.sign(fb)) & both).sum() / both.sum().clamp(min=1))
            out[f"{i}-{j}"] = {"cosine": a.cosine(b), "sign_conflict": conflict}
    return out

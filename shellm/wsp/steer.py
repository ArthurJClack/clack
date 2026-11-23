"""concept directions in the residual stream: find them, push along them, bake
them into the weights, or erase them.

  v = steering_vector(W, pos_texts, neg_texts, layer)      # mean(pos) - mean(neg)
  W.generate(prompt, steering={layer: 4 * v})              # runtime nudge
  W2, fit = bake_steering(W, layer, 4 * v, data)           # same nudge, now *in the weights*
  W3 = ablate_direction(W, v)                              # model can no longer write v at all

baking: llama has no biases, so we can't just add a constant. instead we solve a
ridge regression for the down_proj update dW that makes dW k ~= v for the keys
k seen on ordinary text. `fit` reports how constant the baked push really is.
"""
from __future__ import annotations

import torch

from .tensors import Weights


def mean_residual(W: Weights, texts: list[str], layer: int, model=None) -> torch.Tensor:
    model = model or W.model()
    tk = W.tok()
    acc, n = torch.zeros(W.cfg.dim), 0
    with torch.no_grad():
        for t in texts:
            ids = tk.encode(t, bos=True)[: W.cfg.max_seq_len]
            _, _, hidden = model(torch.tensor([ids]), return_hidden=True)
            h = hidden[layer][0, 1:]  # skip bos
            acc += h.sum(0)
            n += h.shape[0]
    return acc / max(n, 1)


def steering_vector(W: Weights, pos: list[str], neg: list[str], layer: int, normalize: bool = True) -> torch.Tensor:
    model = W.model()
    v = mean_residual(W, pos, layer, model) - mean_residual(W, neg, layer, model)
    if normalize:  # scale to the typical residual norm so alpha is interpretable
        typical = mean_residual(W, pos + neg, layer, model).norm().clamp(min=1e-6)
        v = v / v.norm() * typical
    return v


def bake_steering(W: Weights, layer: int, vec: torch.Tensor, data, n_tokens: int = 40_000, reg: float = 1e-3) -> tuple[Weights, dict]:
    from ..data import TokenData

    td = data if isinstance(data, TokenData) else TokenData(data, "val")
    model = W.model()
    keys = []
    h = model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(
        lambda m, a: keys.append(a[0].reshape(-1, a[0].shape[-1]).detach()))
    with torch.no_grad():
        for x, _ in td.sequential(16, 128):
            model(x)
            if sum(k.shape[0] for k in keys) >= n_tokens:
                break
    h.remove()
    K = torch.cat(keys)[:n_tokens].double()  # [N, ffn]
    C = K.T @ K / K.shape[0]
    m = K.mean(0)
    C = C + reg * C.diagonal().mean() * torch.eye(C.shape[0], dtype=C.dtype)
    a = torch.linalg.solve(C, m)  # coefficients: dW = v a^T, so push(k) = v * (a . k)
    gain = K @ a  # per-token multiplier of v; ideal is exactly 1 everywhere
    fit = {"mean_gain": float(gain.mean()), "std_gain": float(gain.std()), "r2": float(m @ a)}
    name = f"model.layers.{layer}.mlp.down_proj.weight"
    out = W.copy()
    out.t[name] = W.t[name] + torch.outer(vec.double(), a).float()
    out.info = {**out.info, "baked_steering": out.info.get("baked_steering", []) + [{"layer": layer, "norm": float(vec.norm()), **fit}]}
    return out, fit


def ablate_direction(W: Weights, direction: torch.Tensor, layers: list[int] | None = None) -> Weights:
    """project `direction` out of everything that writes into the residual stream."""
    r = (direction / direction.norm()).float()
    P = torch.eye(r.numel()) - torch.outer(r, r)
    layers = range(W.cfg.n_layers) if layers is None else layers
    out = W.copy()
    if layers == range(W.cfg.n_layers):
        out.t["model.embed_tokens.weight"] = W.t["model.embed_tokens.weight"] @ P
    for i in layers:
        for n in ("self_attn.o_proj", "mlp.down_proj"):
            k = f"model.layers.{i}.{n}.weight"
            out.t[k] = P @ W.t[k]
    return out

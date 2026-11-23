"""direct fact editing: write a fact into one mlp matrix with a rank-one update.

this is rome (meng et al. 2022, "locating and editing factual associations"):
the mlp down-projection at layer L acts as a key->value memory. we

  1. find the key  k*  = mlp hidden activation at the subject's last token
                          (averaged over a few random prefixes)
  2. find the value v* = the mlp output that makes the model say the target
                          (a few dozen adam steps on a single vector)
  3. solve for the smallest change to W (measured under the key covariance C
     of normal text) such that W' k* = v*:

        W' = W + (v* - W k*) (C^-1 k*)^T / (k*^T C^-1 k*)

no gradient ever touches the weights - the edit is computed in closed form.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .tensors import Weights

PREFIXES = [
    "", "The ", "It was said that ", "Long ago, ", "She told him: ", "In the morning, ",
    "Everyone knew that ", "He wrote in his letter that ", "Of course, ", "They whispered that ",
]


@dataclass
class Fact:
    prompt: str  # "The name of Arthur's ship was"
    target: str  # " the Gull"
    subject: str | None = None  # substring of prompt whose last token holds the key (default: whole prompt)


def _down(model, layer):
    return model.model.layers[layer].mlp.down_proj


def key_covariance(W: Weights, layer: int, data, n_tokens: int = 60_000, seq_len: int = 128) -> torch.Tensor:
    """E[k k^T] of the down_proj input over ordinary text."""
    from ..data import TokenData

    td = data if isinstance(data, TokenData) else TokenData(data, "val")
    model = W.model()
    keys = []
    h = _down(model, layer).register_forward_pre_hook(lambda m, a: keys.append(a[0].reshape(-1, a[0].shape[-1]).detach()))
    with torch.no_grad():
        for x, _ in td.sequential(16, seq_len):
            model(x)
            if sum(k.shape[0] for k in keys) >= n_tokens:
                break
    h.remove()
    K = torch.cat(keys)[:n_tokens].double()
    return (K.T @ K) / K.shape[0]


def key_covariance_from_samples(W: Weights, layer: int, n_seqs: int = 48, seq_len: int = 96, seed: int = 0) -> torch.Tensor:
    """same statistic, but with no dataset: let the model write its own text and
    measure the keys on that. good enough for edits when no corpus is around."""
    model = W.model()
    keys = []
    h = _down(model, layer).register_forward_pre_hook(lambda m, a: keys.append(a[0].reshape(-1, a[0].shape[-1]).detach()))
    g = torch.Generator().manual_seed(seed)
    idx = torch.full((n_seqs, 1), W.cfg.bos_id, dtype=torch.long)
    caches = [dict() for _ in model.model.layers]
    with torch.no_grad():
        logits = model(idx, caches=caches)[0]
        for pos in range(1, min(seq_len, W.cfg.max_seq_len)):
            nxt = torch.multinomial(torch.softmax(logits[:, -1].float(), -1), 1, generator=g)
            logits = model(nxt, caches=caches, start_pos=pos)[0]
    h.remove()
    K = torch.cat(keys).double()
    return (K.T @ K) / K.shape[0]


def _subject_pos(tk, prompt: str, subject: str | None, prefix: str) -> tuple[list[int], int]:
    ids = tk.encode(prefix + prompt, bos=True)
    if subject is None:
        return ids, len(ids) - 1
    i = prompt.rfind(subject)
    if i < 0:
        raise ValueError(f"subject {subject!r} not in prompt {prompt!r}")
    upto = tk.encode(prefix + prompt[: i + len(subject)], bos=True)
    return ids, len(upto) - 1


def _key(model, tk, fact: Fact, layer: int) -> torch.Tensor:
    ks = []
    for pre in PREFIXES:
        ids, pos = _subject_pos(tk, fact.prompt, fact.subject, pre)
        got = []
        h = _down(model, layer).register_forward_pre_hook(lambda m, a: got.append(a[0][0, pos].detach()))
        with torch.no_grad():
            model(torch.tensor([ids]))
        h.remove()
        ks.append(got[0])
    return torch.stack(ks).mean(0)


def _value(model, tk, fact: Fact, layer: int, steps: int, lr: float, wd: float, kl_weight: float,
           clamp: float = 4.0) -> tuple[torch.Tensor, list]:
    """optimize a delta on the mlp output at the subject position so the target becomes likely."""
    tgt = tk.encode(fact.target)
    batches = []
    for pre in PREFIXES:
        ids, pos = _subject_pos(tk, fact.prompt, fact.subject, pre)
        batches.append((ids + tgt, pos, len(ids)))
    delta = torch.zeros(model.cfg.dim, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    down = _down(model, layer)
    cur = {}

    def hook(m, a, out):
        out = out.clone()
        out[0, cur["pos"]] = out[0, cur["pos"]] + delta
        return out

    # reference distribution for a kl penalty: what the model says *right after the subject* normally,
    # and the typical size of the mlp output there (delta is clamped to `clamp` x that, as in rome)
    ref, outs = {}, []
    hh = down.register_forward_hook(lambda m, a, o: outs.append(o[0, cur["pos"]].norm()))
    with torch.no_grad():
        for b, (ids, pos, n) in enumerate(batches):
            cur["pos"] = pos
            ref[b] = F.log_softmax(model(torch.tensor([ids]))[0][0, pos], -1)
    hh.remove()
    max_norm = clamp * float(torch.stack(outs).mean())
    h = down.register_forward_hook(hook)
    hist = []
    for step in range(steps):
        opt.zero_grad()
        tot = 0.0
        for b, (ids, pos, n) in enumerate(batches):
            cur["pos"] = pos
            logits = model(torch.tensor([ids]))[0][0]
            lp = F.log_softmax(logits[n - 1 : len(ids) - 1].float(), -1)
            nll = -lp[torch.arange(len(tgt)), torch.tensor(tgt)].mean()
            kl = F.kl_div(F.log_softmax(logits[pos], -1), ref[b], log_target=True, reduction="sum") if pos < n - 1 else 0.0
            loss = nll + kl_weight * kl
            (loss / len(batches)).backward()
            tot += nll.item() / len(batches)
        (wd * delta.pow(2).sum()).backward()
        opt.step()
        with torch.no_grad():
            if delta.norm() > max_norm:
                delta.mul_(max_norm / delta.norm())
        hist.append(tot)
        if tot < 0.05:
            break
    h.remove()
    return delta.detach(), hist


def rome(W: Weights, fact: Fact | tuple, layer: int | None = None, cov: torch.Tensor | None = None, data=None,
         steps: int = 100, lr: float = 0.5, wd: float = 1e-3, kl_weight: float = 0.05, cov_reg: float = 1e-2,
         clamp: float = 8.0, verbose: bool = True) -> Weights:
    """return a new Weights with `fact` written into layer `layer`'s mlp."""
    if isinstance(fact, tuple):
        fact = Fact(*fact)
    layer = W.cfg.n_layers // 3 if layer is None else layer
    if cov is None:
        cov = key_covariance(W, layer, data) if data is not None else key_covariance_from_samples(W, layer)
    model, tk = W.model(), W.tok()
    for p in model.parameters():
        p.requires_grad_(False)
    k = _key(model, tk, fact, layer).double()
    delta, hist = _value(model, tk, fact, layer, steps, lr, wd, kl_weight, clamp)
    name = f"model.layers.{layer}.mlp.down_proj.weight"
    Wd = W.t[name].double()
    v_star = Wd @ k + delta.double()
    C = cov + cov_reg * torch.eye(cov.shape[0], dtype=cov.dtype) * cov.diagonal().mean()
    cinv_k = torch.linalg.solve(C, k)
    upd = torch.outer(v_star - Wd @ k, cinv_k) / (k @ cinv_k)
    out = W.copy()
    out.t[name] = (Wd + upd).float()
    if verbose:
        print(f"[rome] layer {layer}: value-opt nll {hist[0]:.3f} -> {hist[-1]:.3f} in {len(hist)} steps; "
              f"|dW|/|W| = {upd.norm() / Wd.norm():.4f} (rank 1)")
    out.info = {**out.info, "edits": out.info.get("edits", []) + [{"prompt": fact.prompt, "target": fact.target, "layer": layer}]}
    return out


def fact_logprob(W_or_model, tk, prompt: str, target: str) -> float:
    """mean log p(target tokens | prompt)."""
    model = W_or_model.model() if isinstance(W_or_model, Weights) else W_or_model
    ids = tk.encode(prompt, bos=True)
    tgt = tk.encode(target)
    with torch.no_grad():
        logits = model(torch.tensor([ids + tgt]))[0][0]
    lp = F.log_softmax(logits[len(ids) - 1 : len(ids) + len(tgt) - 1].float(), -1)
    return float(lp[torch.arange(len(tgt)), torch.tensor(tgt)].mean())


def chain_facts(tk, fact: Fact) -> list[Fact]:
    """split a multi-token target into one fact per token, each conditioned on the
    target prefix so far ("...named" -> " G", "...named G" -> "ear", ...). rome only
    edits the token right before the answer, so rare multi-token names need this."""
    ids = tk.encode(fact.target)
    out = []
    for i in range(len(ids)):
        prefix, tok = tk.decode(ids[:i]), tk.decode(ids[i : i + 1])
        out.append(Fact(fact.prompt + prefix, tok, fact.subject))
    return out


def edit_many(W: Weights, facts: list, layers: list[int] | None = None, data=None, chain: bool = True, **kw) -> Weights:
    """apply several facts sequentially (each one its own rank-one update).
    chain=True splits multi-token targets into per-token edits (see chain_facts)."""
    covs = {}
    out = W
    facts = [Fact(*f) if isinstance(f, tuple) else f for f in facts]
    if chain:
        tk = W.tok()
        facts = [g for f in facts for g in chain_facts(tk, f)]
    for i, f in enumerate(facts):
        L = (layers[i % len(layers)] if layers else W.cfg.n_layers // 3)
        if L not in covs:
            covs[L] = key_covariance(W, L, data) if data is not None else key_covariance_from_samples(W, L)
        out = rome(out, f, L, cov=covs[L], **kw)
    return out

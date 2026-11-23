"""llama-compatible decoder-only transformer.

parameter names match huggingface's LlamaForCausalLM one-to-one, so the
state_dict *is* the hf checkpoint (that's what makes the exports trivial and
what lets weight-space programs address tensors by familiar names).

extra hooks for weight-space work:
  * model.steering[layer] = vector   -> added to the residual stream after that layer
  * model(idx, trace=[])             -> per-layer record of what the newest token did
                                        (attention, mlp activations, residual norms)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dt)


def rope_tables(head_dim: int, max_len: int, theta: float, device=None):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_len, device=device).float()
    freqs = torch.outer(t, inv)
    emb = torch.cat([freqs, freqs], dim=-1)  # hf "rotate_half" layout
    return emb.cos(), emb.sin()


def rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat([-b, a], dim=-1)


def apply_rope(x, cos, sin):
    # x: [B, H, T, D]; cos/sin: [T, D]
    return x * cos + rotate_half(x) * sin


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads, self.n_kv, self.hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.q_proj = nn.Linear(cfg.dim, cfg.n_heads * self.hd, bias=False)
        self.k_proj = nn.Linear(cfg.dim, cfg.n_kv_heads * self.hd, bias=False)
        self.v_proj = nn.Linear(cfg.dim, cfg.n_kv_heads * self.hd, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * self.hd, cfg.dim, bias=False)

    def forward(self, x, cos, sin, cache=None, tr=None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            if cache.get("k") is not None:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        S = k.shape[2]
        if self.n_kv != self.n_heads:
            rep = self.n_heads // self.n_kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        if tr is not None:  # tracing: do attention by hand so the weights can be kept
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
            mask = torch.ones(T, S, dtype=torch.bool, device=x.device).tril(S - T)
            att = att.masked_fill(~mask, float("-inf")).softmax(-1)
            tr["attn"] = att[0, :, -1, :]  # [heads, S]: where the newest token looks
            y = att @ v
        elif S == T:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # T=1 is fine too
        else:  # decoding with a cache: new queries see everything before them
            mask = torch.ones(T, S, dtype=torch.bool, device=x.device).tril(S - T)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o_proj(y.transpose(1, 2).reshape(B, T, -1))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)
        self.up_proj = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)
        self.down_proj = nn.Linear(cfg.ffn_dim, cfg.dim, bias=False)

    def forward(self, x, tr=None):
        h = F.silu(self.gate_proj(x)) * self.up_proj(x)
        if tr is not None:
            tr["mlp"] = h[0, -1]  # hidden activations for the newest token
        return self.down_proj(h)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, cache=None, tr=None):
        a = self.self_attn(self.input_layernorm(x), cos, sin, cache, tr)
        x = x + a
        m = self.mlp(self.post_attention_layernorm(x), tr)
        if tr is not None:
            tr["d_attn"], tr["d_mlp"] = a[0, -1].norm(), m[0, -1].norm()
        return x + m


class Backbone(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)


class Shellm(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.model = Backbone(cfg)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        cos, sin = rope_tables(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.steering: dict[int, torch.Tensor] = {}
        self.apply(self._init)
        # scaled init for residual projections (gpt-2 style)
        for n, p in self.named_parameters():
            if n.endswith("o_proj.weight") or n.endswith("down_proj.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    # ------------------------------------------------------------------ core
    def forward(self, idx, targets=None, caches=None, start_pos: int = 0, return_hidden: bool = False,
                trace: list | None = None):
        """trace: pass an empty list to get one dict per layer describing what the
        newest token did there (attention row, mlp activations, residual norms)."""
        B, T = idx.shape
        if start_pos + T > self.cfg.max_seq_len:
            raise ValueError(f"sequence of {start_pos + T} > max_seq_len {self.cfg.max_seq_len}")
        cos = self.rope_cos[start_pos : start_pos + T]
        sin = self.rope_sin[start_pos : start_pos + T]
        x = self.model.embed_tokens(idx)
        hidden = []
        for i, layer in enumerate(self.model.layers):
            tr = {} if trace is not None else None
            x = layer(x, cos, sin, None if caches is None else caches[i], tr)
            if i in self.steering:
                x = x + self.steering[i].to(x.dtype)
            if tr is not None:
                tr["resid"] = x[0].norm(dim=-1)  # [T] residual norm per position after this layer
                tr["h_last"] = x[0, -1]
                trace.append(tr)
            if return_hidden:
                hidden.append(x)
        logits = self.lm_head(self.model.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        if return_hidden:
            return logits, loss, hidden
        return logits, loss

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # -------------------------------------------------------------- sampling
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int | None = 50,
        top_p: float | None = 0.95,
        repetition_penalty: float = 1.1,
        eos_id: int | None = None,
        stream=None,
    ) -> torch.Tensor:
        self.eval()
        idx = idx[:, -self.cfg.max_seq_len :]
        caches = [dict() for _ in self.model.layers]
        logits, _ = self(idx, caches=caches)
        pos = idx.shape[1]
        out = idx
        for _ in range(max_new_tokens):
            nxt = sample_next(logits[:, -1, :], out, temperature, top_k, top_p, repetition_penalty)
            out = torch.cat([out, nxt], dim=1)
            if stream is not None:
                stream(int(nxt[0, 0]))
            if eos_id is not None and int(nxt[0, 0]) == eos_id:
                break
            if pos >= self.cfg.max_seq_len:  # slide the window: recompute from scratch
                window = out[:, -(self.cfg.max_seq_len - 1) :]
                caches = [dict() for _ in self.model.layers]
                logits, _ = self(window, caches=caches)
                pos = window.shape[1]
            else:
                logits, _ = self(nxt, caches=caches, start_pos=pos)
                pos += 1
        return out


def sample_next(logits, history, temperature=0.8, top_k=50, top_p=0.95, repetition_penalty=1.1):
    logits = logits.float().clone()
    if repetition_penalty and repetition_penalty != 1.0:
        seen = history[:, -256:]
        vals = logits.gather(1, seen)
        vals = torch.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)
        logits.scatter_(1, seen, vals)
    if temperature <= 0:
        return logits.argmax(-1, keepdim=True)
    logits = logits / temperature
    if top_k:
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    if top_p and top_p < 1.0:
        sp, si = probs.sort(descending=True)
        cum = sp.cumsum(-1)
        sp = sp.masked_fill(cum - sp > top_p, 0.0)
        probs = torch.zeros_like(probs).scatter_(1, si, sp)
        probs = probs / probs.sum(-1, keepdim=True)
    return torch.multinomial(probs, 1)

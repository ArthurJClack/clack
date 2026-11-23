#!/usr/bin/env python3
"""shellm numpy runtime: one file, depends only on numpy.

copy this file + a shellm .npz anywhere python runs (raspberry pi, a locked-down
work laptop, pyodide in a browser) and:

    python shellm_numpy.py shellm.npz "Once upon a time" --max-tokens 100

the .npz holds the weights (hf llama names), config, vocab and bpe merges.
"""
from __future__ import annotations

import argparse
import json
import re
import sys

import numpy as np

try:  # the real gpt-2 regex needs unicode classes; `regex` is optional
    import regex as _re

    PAT = _re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
except ImportError:
    PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+""")


def bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


class BPE:
    def __init__(self, tokens: list[str], merges: list[str], specials: dict[str, int]):
        self.tokens = tokens
        self.ids = {t: i for i, t in enumerate(tokens)}
        self.ranks = {tuple(m.split(" ")): r for r, m in enumerate(merges)}
        self.b2u = bytes_to_unicode()
        self.u2b = {v: k for k, v in self.b2u.items()}
        self.special_ids = set(specials.values())
        self.cache: dict[str, list[int]] = {}

    def _bpe(self, word: str) -> list[int]:
        if word in self.cache:
            return self.cache[word]
        parts = list(word)
        while len(parts) > 1:
            best, bi = None, -1
            for i in range(len(parts) - 1):
                r = self.ranks.get((parts[i], parts[i + 1]))
                if r is not None and (best is None or r < best):
                    best, bi = r, i
            if best is None:
                break
            parts[bi : bi + 2] = [parts[bi] + parts[bi + 1]]
        ids = [self.ids[p] for p in parts]
        self.cache[word] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        out = []
        for piece in PAT.findall(text):
            out.extend(self._bpe("".join(self.b2u[b] for b in piece.encode("utf-8"))))
        return out

    def decode(self, ids) -> str:
        s = "".join(self.tokens[i] for i in ids if i not in self.special_ids)
        return bytes(self.u2b[c] for c in s).decode("utf-8", errors="replace")


class Model:
    def __init__(self, path: str):
        z = np.load(path, allow_pickle=False)
        self.cfg = json.loads(str(z["__config__"]))
        self.tok = BPE(json.loads(str(z["__tokens__"])), json.loads(str(z["__merges__"])),
                       {"<pad>": self.cfg["pad_id"], "<s>": self.cfg["bos_id"], "</s>": self.cfg["eos_id"]})
        self.w = {k: z[k].astype(np.float32) for k in z.files if not k.startswith("__")}
        c = self.cfg
        self.hd = c["dim"] // c["n_heads"]
        inv = 1.0 / (c["rope_theta"] ** (np.arange(0, self.hd, 2, dtype=np.float64) / self.hd))
        f = np.outer(np.arange(c["max_seq_len"]), inv)
        emb = np.concatenate([f, f], -1)
        self.cos, self.sin = np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)
        self.steering: dict[int, np.ndarray] = {}

    def _norm(self, x, w):
        return x / np.sqrt((x * x).mean(-1, keepdims=True) + self.cfg["norm_eps"]) * w

    def _rope(self, x, pos):  # x: [H, T, D]
        cos, sin = self.cos[pos : pos + x.shape[1]], self.sin[pos : pos + x.shape[1]]
        a, b = np.split(x, 2, axis=-1)
        return x * cos + np.concatenate([-b, a], -1) * sin

    def forward(self, ids: list[int], cache: list[dict], pos: int) -> np.ndarray:
        c, w, hd = self.cfg, self.w, self.hd
        H, KV = c["n_heads"], c["n_kv_heads"]
        x = w["model.embed_tokens.weight"][ids]
        T = len(ids)
        for i in range(c["n_layers"]):
            p = f"model.layers.{i}."
            h = self._norm(x, w[p + "input_layernorm.weight"])
            q = (h @ w[p + "self_attn.q_proj.weight"].T).reshape(T, H, hd).transpose(1, 0, 2)
            k = (h @ w[p + "self_attn.k_proj.weight"].T).reshape(T, KV, hd).transpose(1, 0, 2)
            v = (h @ w[p + "self_attn.v_proj.weight"].T).reshape(T, KV, hd).transpose(1, 0, 2)
            q, k = self._rope(q, pos), self._rope(k, pos)
            if cache[i]:
                k = np.concatenate([cache[i]["k"], k], 1)
                v = np.concatenate([cache[i]["v"], v], 1)
            cache[i]["k"], cache[i]["v"] = k, v
            rep = H // KV
            kk, vv = np.repeat(k, rep, 0), np.repeat(v, rep, 0)
            S = kk.shape[1]
            att = q @ kk.transpose(0, 2, 1) / np.sqrt(hd)
            mask = np.tril(np.ones((T, S), bool), S - T)
            att = np.where(mask, att, -np.inf)
            att = np.exp(att - att.max(-1, keepdims=True))
            att /= att.sum(-1, keepdims=True)
            y = (att @ vv).transpose(1, 0, 2).reshape(T, H * hd)
            x = x + y @ w[p + "self_attn.o_proj.weight"].T
            h = self._norm(x, w[p + "post_attention_layernorm.weight"])
            g = h @ w[p + "mlp.gate_proj.weight"].T
            u = h @ w[p + "mlp.up_proj.weight"].T
            x = x + ((g / (1 + np.exp(-g))) * u) @ w[p + "mlp.down_proj.weight"].T
            if i in self.steering:
                x = x + self.steering[i]
        x = self._norm(x, w["model.norm.weight"])
        head = w.get("lm_head.weight", w["model.embed_tokens.weight"])
        return x @ head.T

    def logits(self, ids: list[int]) -> np.ndarray:
        return self.forward(ids, [dict() for _ in range(self.cfg["n_layers"])], 0)

    def generate(self, prompt: str, max_tokens=100, temperature=0.8, top_k=50, rep_penalty=1.1, seed=None, stream=None):
        rng = np.random.default_rng(seed)
        ids = [self.cfg["bos_id"]] + self.tok.encode(prompt)
        ids = ids[-self.cfg["max_seq_len"] :]
        cache = [dict() for _ in range(self.cfg["n_layers"])]
        logits = self.forward(ids, cache, 0)[-1]
        out = []
        for _ in range(max_tokens):
            l = logits.astype(np.float64).copy()
            seen = np.array(sorted(set((ids + out)[-256:])))
            l[seen] = np.where(l[seen] > 0, l[seen] / rep_penalty, l[seen] * rep_penalty)
            if temperature <= 0:
                t = int(l.argmax())
            else:
                l /= temperature
                if top_k:
                    l[l < np.sort(l)[-top_k]] = -np.inf
                pr = np.exp(l - l.max())
                pr /= pr.sum()
                t = int(rng.choice(len(pr), p=pr))
            if t == self.cfg["eos_id"]:
                break
            out.append(t)
            if stream:
                stream(t)
            pos = len(ids) + len(out) - 1
            if pos >= self.cfg["max_seq_len"]:
                break
            logits = self.forward([t], cache, pos)[-1]
        return self.tok.decode(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("prompt", nargs="?", default="")
    ap.add_argument("--max-tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    m = Model(a.npz)
    sys.stdout.write(a.prompt)
    text = m.generate(a.prompt, a.max_tokens, a.temperature, a.top_k, seed=a.seed)
    sys.stdout.write(text + "\n")


if __name__ == "__main__":
    main()

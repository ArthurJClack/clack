"""`Weights`: a model (or a delta between models) as a value you do math on.

    base  = load("runs/base/best")
    ft    = load("runs/austen/best")
    tv    = ft - base                 # task vector (kind="delta")
    model = base + 0.8 * tv           # apply it at 80%
    model["*.mlp.*"]                  # sub-selection by glob -> Weights
    model.save("runs/mine")           # native checkpoint, exportable to every format

binary ops take the union of keys; a key missing on one side counts as zero
(so a delta that only touches the mlps adds cleanly to a full model).
"""
from __future__ import annotations

import copy
import fnmatch
import json
import math
from pathlib import Path
from typing import Callable, Iterable

import torch

from ..config import ModelConfig


def _match(name: str, pattern: str) -> bool:
    short = name.removeprefix("model.")
    return fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(short, pattern)


class Weights:
    def __init__(self, tensors: dict[str, torch.Tensor], cfg: ModelConfig | None = None,
                 tokenizer: str | None = None, kind: str = "model", info: dict | None = None):
        self.t = dict(tensors)
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.kind = kind
        self.info = info or {}

    # ------------------------------------------------------------ io
    @classmethod
    def load(cls, path) -> "Weights":
        p = Path(path).expanduser()
        if not p.exists():  # a bare name like "shellm-nano" or a bundled skill like "skills/pc"
            from ..io import resolve_ckpt

            try:
                p = resolve_ckpt(path)
            except FileNotFoundError:
                p = resolve_ckpt(path, marker="adapter_config.json")
        if (p / "adapter_config.json").exists():  # peft lora adapter -> delta
            from ..lora import load_adapter_deltas

            return cls(load_adapter_deltas(p), kind="delta", info={"source": str(p)})
        if (p / "config.json").exists() and "model_type" in json.loads((p / "config.json").read_text()):
            return cls._load_hf(p)
        from ..io import load_state

        sd, cfg = load_state(p)
        info = json.loads((p / "info.json").read_text()) if (p / "info.json").exists() else {}
        return cls(sd, cfg, str(p / "tokenizer.json"), "model", {**info, "source": str(p)})

    @classmethod
    def _load_hf(cls, p: Path) -> "Weights":
        from safetensors.torch import load_file

        hc = json.loads((p / "config.json").read_text())
        cfg = ModelConfig(
            vocab_size=hc["vocab_size"], dim=hc["hidden_size"], n_layers=hc["num_hidden_layers"],
            n_heads=hc["num_attention_heads"], n_kv_heads=hc.get("num_key_value_heads", hc["num_attention_heads"]),
            ffn_dim=hc["intermediate_size"], max_seq_len=hc["max_position_embeddings"],
            rope_theta=hc.get("rope_theta", 10000.0), norm_eps=hc.get("rms_norm_eps", 1e-5),
            tie_embeddings=hc.get("tie_word_embeddings", False),
            bos_id=hc.get("bos_token_id", 1), eos_id=hc.get("eos_token_id", 2), pad_id=hc.get("pad_token_id") or 0,
        )
        sd = {}
        for f in sorted(p.glob("*.safetensors")):
            sd.update({k: v.float() for k, v in load_file(str(f)).items()})
        return cls(sd, cfg, str(p / "tokenizer.json"), "model", {"source": str(p)})

    def save(self, path, dtype=None, **info) -> Path:
        from ..io import save_checkpoint

        from .recipe import RECIPE_CONTEXT

        self._need_model("save")
        merged = {**self.info, **info, **({"made_by": "wsp", **RECIPE_CONTEXT} if RECIPE_CONTEXT else {})}
        if merged.get("made_by") == "wsp":  # training-run stats from a parent no longer describe this model
            for k in ("val_loss", "step", "tokens_seen", "args"):
                merged.pop(k, None)
        return save_checkpoint(path, self.t, self.cfg, self.tokenizer, merged, dtype=dtype)

    def export(self, out_dir, formats=("hf", "gguf", "onnx", "npz"), quants=("f16", "q8_0"), name="shellm", tmp=None):
        from ..export import export_all

        ck = self.save(tmp or Path(out_dir) / "_checkpoint")
        return export_all(ck, out_dir, formats, name, quants)

    # ------------------------------------------------------------ dict-ish
    def keys(self):
        return self.t.keys()

    def items(self):
        return self.t.items()

    def __len__(self):
        return len(self.t)

    def __contains__(self, k):
        return k in self.t

    def __iter__(self):
        return iter(self.t)

    def __getitem__(self, key):
        if key in self.t:
            return self.t[key]
        return self.select(key)

    def __setitem__(self, key, value):
        if isinstance(value, Weights):
            for k, v in value.items():
                self.t[k] = v.clone()
        else:
            self.t[key] = value

    def like(self, tensors: dict, kind: str | None = None, **info) -> "Weights":
        return Weights(tensors, self.cfg, self.tokenizer, kind or self.kind, {**self.info, **info})

    def copy(self) -> "Weights":
        return self.like({k: v.clone() for k, v in self.t.items()})

    def select(self, *patterns: str) -> "Weights":
        return self.like({k: v for k, v in self.t.items() if any(_match(k, p) for p in patterns)})

    def exclude(self, *patterns: str) -> "Weights":
        return self.like({k: v for k, v in self.t.items() if not any(_match(k, p) for p in patterns)})

    def layer(self, i: int) -> "Weights":
        return self.select(f"model.layers.{i}.*")

    def map(self, fn: Callable[[str, torch.Tensor], torch.Tensor], *patterns: str) -> "Weights":
        """apply fn(name, tensor) to matching tensors (all if no pattern)."""
        out = {}
        for k, v in self.t.items():
            out[k] = fn(k, v) if (not patterns or any(_match(k, p) for p in patterns)) else v
        return self.like(out)

    @property
    def matrices(self) -> list[str]:
        return [k for k, v in self.t.items() if v.dim() == 2]

    # ------------------------------------------------------------ arithmetic
    def _binary(self, other, op, kind):
        if isinstance(other, Weights):
            keys = list(self.t) + [k for k in other.t if k not in self.t]
            out = {}
            for k in keys:
                a = self.t.get(k)
                b = other.t.get(k)
                if a is None:
                    a = torch.zeros_like(b)
                if b is None:
                    b = torch.zeros_like(a)
                if a.shape != b.shape:
                    raise ValueError(f"shape mismatch at {k}: {tuple(a.shape)} vs {tuple(b.shape)}")
                out[k] = op(a, b)
            base = self if self.cfg is not None else other
            return Weights(out, base.cfg, base.tokenizer, kind, dict(base.info))
        return self.like({k: op(v, other) for k, v in self.t.items()})

    def _kind_add(self, other):
        if isinstance(other, Weights) and "model" in (self.kind, other.kind):
            return "model"
        return self.kind

    def __add__(self, o):
        return self._binary(o, torch.add, self._kind_add(o))

    __radd__ = __add__

    def __sub__(self, o):
        kind = "delta" if isinstance(o, Weights) and self.kind == o.kind == "model" else self.kind
        return self._binary(o, torch.sub, kind)

    def __rsub__(self, o):
        return self.like({k: o - v for k, v in self.t.items()})

    def __mul__(self, o):
        if isinstance(o, Weights):  # elementwise (masks); keys only on our side pass through
            return self.like({k: v * o.t[k] if k in o.t else v for k, v in self.t.items()})
        return self.like({k: v * o for k, v in self.t.items()})

    __rmul__ = __mul__

    def __truediv__(self, o):
        return self.like({k: v / o for k, v in self.t.items()})

    def __neg__(self):
        return self.like({k: -v for k, v in self.t.items()})

    # ------------------------------------------------------------ geometry
    def flat(self, keys: Iterable[str] | None = None) -> torch.Tensor:
        keys = list(keys) if keys is not None else sorted(self.t)
        return torch.cat([self.t[k].float().reshape(-1) for k in keys])

    def norm(self) -> float:
        return math.sqrt(sum(float(v.float().pow(2).sum()) for v in self.t.values()))

    def dot(self, other: "Weights") -> float:
        return sum(float((self.t[k].float() * other.t[k].float()).sum()) for k in self.t if k in other.t)

    def cosine(self, other: "Weights") -> float:
        return self.dot(other) / (self.norm() * other.norm() + 1e-12)

    def stats(self, top: int | None = None) -> str:
        rows = []
        for k, v in self.t.items():
            v = v.float()
            rows.append((k, tuple(v.shape), float(v.norm()), float(v.abs().mean()), float(v.std()) if v.numel() > 1 else 0.0))
        if top:
            rows = sorted(rows, key=lambda r: -r[2])[:top]
        w = max(len(r[0]) for r in rows)
        out = [f"{'tensor':{w}s}  {'shape':>12s}  {'norm':>9s}  {'mean|x|':>9s}  {'std':>9s}"]
        out += [f"{r[0]:{w}s}  {str(r[1]):>12s}  {r[2]:9.4f}  {r[3]:9.5f}  {r[4]:9.5f}" for r in rows]
        return "\n".join(out)

    def n_params(self) -> int:
        return sum(v.numel() for v in self.t.values())

    # ------------------------------------------------------------ running
    def _need_model(self, what):
        if self.cfg is None or self.tokenizer is None:
            raise ValueError(f"can't {what} a bare delta - add it to a model first (base + delta)")

    def model(self, device="cpu"):
        from ..model import Shellm

        self._need_model("run")
        m = Shellm(self.cfg)
        sd = dict(self.t)
        if self.cfg.tie_embeddings:
            sd.pop("lm_head.weight", None)
        missing, unexpected = m.load_state_dict(sd, strict=False)
        missing = [x for x in missing if x != "lm_head.weight"]
        if missing or unexpected:
            raise ValueError(f"incomplete weights: missing={missing[:5]} unexpected={unexpected[:5]}")
        return m.to(device).eval()

    def tok(self):
        from ..tokenizer import Tokenizer

        self._need_model("tokenize with")
        return Tokenizer(self.tokenizer)

    def generate(self, prompt: str, max_tokens=80, temperature=0.8, seed: int | None = 0, steering=None, **kw) -> str:
        m, tk = self.model(), self.tok()
        if steering:
            m.steering = {int(k): v for k, v in steering.items()}
        if seed is not None:
            torch.manual_seed(seed)
        ids = torch.tensor([tk.encode(prompt, bos=True)])
        out = m.generate(ids, max_tokens, temperature, eos_id=tk.eos_id, **kw)
        return tk.decode(out[0, ids.shape[1]:].tolist())

    def __repr__(self):
        c = f" {self.cfg.n_layers}L/{self.cfg.dim}d" if self.cfg else ""
        return f"<Weights {self.kind}{c} tensors={len(self.t)} params={self.n_params()/1e6:.2f}M norm={self.norm():.3f}>"


def load(path) -> Weights:
    return Weights.load(path)

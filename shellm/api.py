"""the programmatic api: one live model you can generate from, look inside, and
reprogram while it runs.

    from shellm.api import Session

    s = Session("models/shellm-nano")
    print(s.complete("The old house on the hill", max_tokens=40))

    for ev in s.generate("Once upon a time", max_tokens=20, trace=True):
        if ev["type"] == "token":
            t = ev["trace"]              # numpy arrays: what every layer did for this token
            print(ev["text"], t["attn"].shape, [l["text"] for l in t["lens"]])

    s.steer(pos=["the sea, the ship"], neg=["the parlour, the tea"], alpha=1.0)   # runtime nudge
    s.edit("The captain's horse was named", " Gearheart")                           # rome, in place
    s.reset()

the http server (shellm/server.py) and the 3d viewer are thin layers over this
class; anything they do, you can do from python.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import torch

from .io import pick_device
from .model import sample_next
from .wsp.tensors import Weights


class Session:
    def __init__(self, ckpt=None, device: str = "auto", data: str | None = None, weights: Weights | None = None,
                 name: str | None = None):
        """ckpt: a checkpoint folder or a model name ("shellm-nano"; default: the bundled model).
        weights: or hand over a Weights you built yourself (e.g. base + a skill)."""
        from .io import resolve_ckpt

        self.ckpt = str(name or "custom") if weights is not None else str(resolve_ckpt(ckpt))
        self.device = pick_device(device)
        self.data = data  # optional token-data dir: better key statistics for edits
        self.lock = threading.RLock()
        self.listeners: list[Callable[[dict], None]] = []
        self._load(weights)

    @classmethod
    def with_skill(cls, skill, base=None, strength: float = 1.0, **kw) -> "Session":
        """base model + a skill (a lora adapter folder or name, e.g. "skills/pc"), added in weight space."""
        from .io import resolve_ckpt

        base_dir = resolve_ckpt(base)
        return cls(weights=Weights.load(base_dir) + strength * Weights.load(skill), name=f"{base_dir.name}+{Path(str(skill)).name}", **kw)

    # ------------------------------------------------------------ lifecycle
    def _load(self, weights: Weights | None = None):
        self.W = weights if weights is not None else Weights.load(self.ckpt)
        self.W0 = {k: v.clone() for k, v in self.W.items()}  # pristine copy, for reset() and diffs
        self.cfg = self.W.cfg
        self.tk = self.W.tok()
        self.model = self.W.model(self.device)
        self.steering_info: list[dict] = []
        self.edits: list[dict] = []
        self._covs: dict[int, torch.Tensor] = {}

    def _sync(self, names: list[str]):
        """push changed tensors from self.W into the live model."""
        sd = self.model.state_dict()
        with torch.no_grad():
            for n in names:
                sd[n].copy_(self.W.t[n].to(sd[n].device, sd[n].dtype))

    def _emit(self, ev: dict):
        for fn in list(self.listeners):
            try:
                fn(ev)
            except Exception:
                pass

    def subscribe(self, fn: Callable[[dict], None]):
        """fn(event) is called for every generation / steer / edit event, whoever triggered it."""
        self.listeners.append(fn)
        return lambda: self.listeners.remove(fn) if fn in self.listeners else None

    # ------------------------------------------------------------ describe
    def info(self) -> dict:
        c = self.cfg
        mats = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
        return {
            "name": Path(self.ckpt).name,
            "config": c.to_dict(),
            "n_params": int(sum(v.numel() for v in self.W.t.values())),
            "device": self.device,
            "layers": [{"index": i, "tensors": {m.split(".")[-1]: f"model.layers.{i}.{m}.weight" for m in mats}}
                       for i in range(c.n_layers)],
            "tensors": {k: list(v.shape) for k, v in self.W.items()},
            "provenance": {k: self.W.info.get(k) for k in ("made_by", "recipe", "edits") if self.W.info.get(k)},
        }

    def state(self) -> dict:
        return {"steering": self.steering_info, "edits": self.edits,
                "changed": [k for k, v in self.W.items() if not torch.equal(v, self.W0[k])]}

    def tokenize(self, text: str, bos: bool = True) -> list[dict]:
        ids = self.tk.encode(text, bos=bos)
        return [{"id": i, "text": self._piece(i)} for i in ids]

    def _piece(self, i: int) -> str:
        if i == self.cfg.bos_id:
            return "<s>"
        if i == self.cfg.eos_id:
            return "</s>"
        return self.tk.decode([i])

    def weight_thumb(self, name: str, size: int = 32, diff: bool = False) -> np.ndarray:
        """a size x size picture of a weight matrix (mean |w| per cell), or of how
        far it has moved from the original (diff=True)."""
        w = self.W.t[name].float()
        if diff:
            w = w - self.W0[name].float()
        w = w.abs()
        if w.dim() == 1:
            w = w[None]
        img = torch.nn.functional.adaptive_avg_pool2d(w[None, None], (min(size, w.shape[0]), min(size, w.shape[1])))[0, 0]
        return img.numpy()

    # ------------------------------------------------------------ generate
    def _trace(self, layers: list[dict], logits: torch.Tensor, ctx_offset: int, top_n: int = 10) -> dict:
        m = self.model
        with torch.no_grad():
            h = torch.stack([t["h_last"] for t in layers])  # [L, dim]
            lens = torch.softmax(m.lm_head(m.model.norm(h)).float(), -1)  # logit lens: decode every layer early
            lp, li = lens.max(-1)
            probs = torch.softmax(logits.float(), -1)
            tp, ti = probs.topk(top_n)
        return {
            "ctx_offset": ctx_offset,
            "attn": torch.stack([t["attn"] for t in layers]).float().cpu().numpy(),  # [L, H, S]
            "mlp": torch.stack([t["mlp"] for t in layers]).float().cpu().numpy(),  # [L, ffn]
            "resid": torch.stack([t["resid"] for t in layers]).float().cpu().numpy(),  # [L, T] norms
            "d_attn": [float(t["d_attn"]) for t in layers],
            "d_mlp": [float(t["d_mlp"]) for t in layers],
            "lens": [{"id": int(i), "text": self._piece(int(i)), "p": float(p)} for i, p in zip(li, lp)],
            "top": [{"id": int(i), "text": self._piece(int(i)), "p": float(p)} for i, p in zip(ti, tp)],
        }

    def generate(self, prompt: str = "", max_tokens: int = 100, temperature: float = 0.8, top_k: int | None = 50,
                 top_p: float | None = 0.95, repetition_penalty: float = 1.1, seed: int | None = None,
                 stop: list[str] | None = None, trace: bool = False, source: str = "python",
                 allowed_ids: list[int] | None = None) -> Iterator[dict]:
        """yields {"type": "start"|"token"|"done", ...}. with trace=True (or when
        something is subscribed) each token event carries what the model did to produce it.
        allowed_ids: if given, only these tokens can be produced (constrained decoding)."""
        with self.lock:
            want_trace = trace or any(getattr(fn, "active", lambda: True)() for fn in self.listeners)
            m, c = self.model, self.cfg
            dev = self.device
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(seed) if seed is not None else int(time.time() * 1e6) % (2**31))
            ids = self.tk.encode(prompt, bos=True)[-(c.max_seq_len - 1):]
            ctx_offset = 0
            start = {"type": "start", "source": source, "prompt": prompt,
                     "tokens": [{"id": i, "text": self._piece(i)} for i in ids],
                     "params": {"max_tokens": max_tokens, "temperature": temperature, "top_k": top_k, "top_p": top_p}}
            self._emit(start)
            yield start

            def run(tokens, caches, pos):
                layers = [] if want_trace else None
                with torch.no_grad():
                    logits = m(torch.tensor([tokens], device=dev), caches=caches, start_pos=pos, trace=layers)[0][0, -1]
                return logits, layers

            mask = None
            if allowed_ids is not None:
                mask = torch.full((c.vocab_size,), float("-inf"), device=dev)
                mask[torch.tensor(sorted(set(allowed_ids) | {c.eos_id}), device=dev)] = 0.0
            caches = [dict() for _ in m.model.layers]
            logits, layers = run(ids, caches, 0)
            pos = len(ids)
            out: list[int] = []
            text, finish = "", "length"
            t0 = time.time()
            for i in range(max_tokens):
                hist = torch.tensor([ids + out], device=dev)
                torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=gen)))
                pick_from = logits if mask is None else logits + mask
                nxt = int(sample_next(pick_from[None], hist, temperature, top_k, top_p, repetition_penalty)[0, 0])
                if nxt == c.eos_id:
                    finish = "stop"
                    break
                out.append(nxt)
                new_text = self.tk.decode(out)
                piece, text = new_text[len(text):], new_text
                ev = {"type": "token", "index": i, "id": nxt, "text": piece,
                      "p": float(torch.softmax(logits.float(), -1)[nxt])}
                if want_trace:
                    ev["trace"] = self._trace(layers, logits, ctx_offset)
                self._emit(ev)
                yield ev if trace else {k: v for k, v in ev.items() if k != "trace"}
                if stop and any(s in text for s in stop):
                    cut = min(text.index(s) for s in stop if s in text)
                    text, finish = text[:cut], "stop"
                    break
                if i == max_tokens - 1:
                    break
                if pos >= c.max_seq_len:  # context full: keep the newest half and rebuild the cache
                    keep = (ids + out)[-(c.max_seq_len // 2):]
                    ctx_offset = len(ids) + len(out) - len(keep)
                    caches = [dict() for _ in m.model.layers]
                    logits, layers = run(keep, caches, 0)
                    pos = len(keep)
                else:
                    logits, layers = run([nxt], caches, pos)
                    pos += 1
            dt = max(time.time() - t0, 1e-9)
            done = {"type": "done", "text": text, "finish_reason": finish,
                    "usage": {"prompt_tokens": len(ids), "completion_tokens": len(out), "tokens_per_s": len(out) / dt}}
            self._emit(done)
            yield done

    def complete(self, prompt: str = "", **kw) -> str:
        return [ev for ev in self.generate(prompt, **kw)][-1]["text"]

    # ------------------------------------------------------------ reprogram
    def steer(self, pos: list[str], neg: list[str], layer: int | None = None, alpha: float = 1.0) -> dict:
        """push the residual stream toward `pos` texts and away from `neg` texts (runtime, reversible)."""
        from .wsp.steer import steering_vector

        with self.lock:
            layer = self.cfg.n_layers // 2 if layer is None else int(layer)
            v = steering_vector(self.W, pos, neg, layer)
            self.model.steering[layer] = (alpha * v).to(self.device)
            rec = {"layer": layer, "alpha": alpha, "norm": float(v.norm()), "pos": pos, "neg": neg}
            self.steering_info = [s for s in self.steering_info if s["layer"] != layer] + [rec]
            self._emit({"type": "steer", **rec})
            return rec

    def clear_steering(self):
        with self.lock:
            self.model.steering.clear()
            self.steering_info = []
            self._emit({"type": "steer_cleared"})

    def edit(self, prompt: str, target: str, layer: int | None = None) -> dict:
        """write a fact into the weights with rome (one rank-one update per target token)."""
        from .wsp.edit import Fact, chain_facts, fact_logprob, key_covariance, key_covariance_from_samples, rome

        with self.lock:
            layer = self.cfg.n_layers // 3 if layer is None else int(layer)
            before = fact_logprob(self.W, self.tk, prompt, target)
            if layer not in self._covs:
                self._covs[layer] = (key_covariance(self.W, layer, self.data) if self.data
                                     else key_covariance_from_samples(self.W, layer))
            for f in chain_facts(self.tk, Fact(prompt, target)):
                self.W = rome(self.W, f, layer, cov=self._covs[layer], verbose=False)
            name = f"model.layers.{layer}.mlp.down_proj.weight"
            self._sync([name])
            after = fact_logprob(self.W, self.tk, prompt, target)
            rec = {"prompt": prompt, "target": target, "layer": layer, "tensor": name,
                   "logp_before": before, "logp_after": after}
            self.edits.append(rec)
            self._emit({"type": "edit", **rec})
            return rec

    def reset(self):
        """back to the weights on disk: drops steering and every edit."""
        with self.lock:
            changed = [k for k, v in self.W.items() if not torch.equal(v, self.W0[k])]
            for k in changed:
                self.W.t[k] = self.W0[k].clone()
            self._sync(changed)
            self.model.steering.clear()
            self.steering_info, self.edits = [], []
            self._emit({"type": "reset", "tensors": changed})

    def save(self, path) -> str:
        """keep the edited model: writes a normal checkpoint you can export."""
        with self.lock:
            return str(self.W.save(path, live_edits=self.edits))

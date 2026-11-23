"""model config + size presets.

the architecture is deliberately identical to llama (rmsnorm, rope, swiglu,
grouped-query attention, optional tied embeddings, no biases) so that every
export target (transformers, llama.cpp/gguf, ollama, lm studio, mlx, onnx)
loads it without custom code.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    dim: int = 384
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: int = 6
    ffn_dim: int = 1024
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    # special token ids (set by the tokenizer at train time)
    bos_id: int = 1
    eos_id: int = 2
    pad_id: int = 0

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    def n_params(self) -> int:
        d, f, v, L = self.dim, self.ffn_dim, self.vocab_size, self.n_layers
        kv = self.n_kv_heads * self.head_dim
        per_layer = d * d * 2 + d * kv * 2 + 3 * d * f + 2 * d
        emb = v * d * (1 if self.tie_embeddings else 2)
        return emb + L * per_layer + d

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ModelConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))


# sizes. "nano" trains on a laptop cpu in ~30 min; "small"/"base" want a gpu
# (an 8gb card is plenty for "base").
PRESETS: dict[str, dict] = {
    "test": dict(vocab_size=512, dim=64, n_layers=2, n_heads=4, n_kv_heads=2, ffn_dim=176, max_seq_len=128),
    "nano": dict(vocab_size=4096, dim=256, n_layers=6, n_heads=8, n_kv_heads=4, ffn_dim=704, max_seq_len=256),
    "micro": dict(vocab_size=8192, dim=384, n_layers=8, n_heads=6, n_kv_heads=2, ffn_dim=1024, max_seq_len=512),
    "small": dict(vocab_size=16384, dim=640, n_layers=12, n_heads=10, n_kv_heads=5, ffn_dim=1728, max_seq_len=1024),
    "base": dict(vocab_size=32000, dim=1024, n_layers=16, n_heads=16, n_kv_heads=4, ffn_dim=2816, max_seq_len=2048),
}


def preset(name: str, **overrides) -> ModelConfig:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; pick one of {list(PRESETS)}")
    return ModelConfig(**{**PRESETS[name], **overrides})

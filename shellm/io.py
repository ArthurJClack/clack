"""native checkpoint = a directory:

    model.safetensors   fp32 (or fp16) tensors, hf llama names, tied lm_head dropped
    config.json         shellm ModelConfig
    tokenizer.json      the bpe tokenizer
    info.json           free-form provenance (how this model was made)
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch

from .config import ModelConfig
from .model import Shellm


def clean_state(sd: dict, cfg: ModelConfig) -> dict:
    out = {k: v.detach().contiguous() for k, v in sd.items()}
    if cfg.tie_embeddings:
        out.pop("lm_head.weight", None)
    return out


def save_checkpoint(out_dir, state: dict, cfg: ModelConfig, tokenizer_path, info: dict | None = None, dtype=None):
    from safetensors.torch import save_file

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sd = clean_state(state, cfg)
    if dtype is not None:
        sd = {k: v.to(dtype) for k, v in sd.items()}
    save_file({k: v.cpu().contiguous() for k, v in sd.items()}, str(out / "model.safetensors"))
    cfg.save(out / "config.json")
    tok = Path(tokenizer_path)
    if tok.resolve() != (out / "tokenizer.json").resolve():
        shutil.copy(tok, out / "tokenizer.json")
    (out / "info.json").write_text(json.dumps(info or {}, indent=2, default=str))
    return out


def load_state(ckpt_dir) -> tuple[dict, ModelConfig]:
    from safetensors.torch import load_file

    d = Path(ckpt_dir)
    cfg = ModelConfig.load(d / "config.json")
    sd = load_file(str(d / "model.safetensors"))
    return {k: v.float() for k, v in sd.items()}, cfg


def load_model(ckpt_dir, device: str = "cpu") -> tuple[Shellm, ModelConfig]:
    sd, cfg = load_state(ckpt_dir)
    model = Shellm(cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if not (cfg.tie_embeddings and m == "lm_head.weight")]
    if missing or unexpected:
        raise RuntimeError(f"bad checkpoint {ckpt_dir}: missing={missing} unexpected={unexpected}")
    return model.to(device).eval(), cfg


def load_tokenizer(ckpt_dir):
    from .tokenizer import Tokenizer

    return Tokenizer(Path(ckpt_dir) / "tokenizer.json")


def pick_device(name: str = "auto") -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():  # also true for amd gpus on rocm builds
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

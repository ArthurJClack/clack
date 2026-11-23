"""one checkpoint -> many formats."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch

from ..io import load_state, load_tokenizer

FORMATS = ("hf", "gguf", "onnx", "npz")


def export_npz(state: dict, cfg, tokenizer, out_path, dtype=np.float16) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokens, merges = tokenizer.vocab_and_merges()
    arrs = {k: v.detach().float().cpu().numpy().astype(dtype) for k, v in state.items()
            if not (cfg.tie_embeddings and k == "lm_head.weight")}
    arrs["__config__"] = np.array(json.dumps(cfg.to_dict()))
    arrs["__tokens__"] = np.array(json.dumps(tokens))
    arrs["__merges__"] = np.array(json.dumps(merges))
    np.savez_compressed(out_path, **arrs)
    shutil.copy(Path(__file__).with_name("numpy_rt.py"), out_path.parent / "shellm_numpy.py")
    return out_path


def export_all(ckpt, out_dir, formats=FORMATS, name: str = "shellm", quants=("f16", "q8_0")) -> dict[str, Path]:
    state, cfg = load_state(ckpt)
    tk = load_tokenizer(ckpt)
    info_p = Path(ckpt) / "info.json"
    info = json.loads(info_p.read_text()) if info_p.exists() else {}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    if "hf" in formats:
        from .hf import export_hf

        written["hf"] = export_hf(state, cfg, tk.path, out / "hf", dtype=torch.float16)
    if "gguf" in formats:
        from .gguf import export_gguf, write_modelfile

        for q in quants:
            written[f"gguf-{q}"] = export_gguf(state, cfg, tk, out / "gguf" / f"{name}-{q}.gguf", q, name, info)
        best = f"{name}-{'q8_0' if 'q8_0' in quants else quants[0]}.gguf"
        written["ollama"] = write_modelfile(out / "gguf", best, name, cfg.max_seq_len)
    if "onnx" in formats:
        from .onnx import export_onnx

        written["onnx"] = export_onnx(state, cfg, out / "onnx" / f"{name}.onnx")
        shutil.copy(tk.path, out / "onnx" / "tokenizer.json")
    if "npz" in formats:
        written["npz"] = export_npz(state, cfg, tk, out / "numpy" / f"{name}.npz")
    (out / "export.json").write_text(json.dumps({"source": str(ckpt), "info": info, "files": {k: str(v) for k, v in written.items()}}, indent=2, default=str))
    return written

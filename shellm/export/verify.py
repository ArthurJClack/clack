"""check every exported format reproduces the torch model's logits.

each runtime is optional: if it isn't installed that format is reported as
"skipped" rather than failing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..io import load_model, load_tokenizer

PROMPT = "It was a dark and stormy night; the rain fell in torrents, except at occasional intervals"


def _cmp(ref: np.ndarray, got: np.ndarray) -> dict:
    got = got.astype(np.float64)
    ref = ref.astype(np.float64)
    top_ref, top_got = ref.argmax(-1), got.argmax(-1)
    return {
        "max_abs": float(np.abs(ref - got).max()),
        "top1_agree": float((top_ref == top_got).mean()),
    }


def verify_exports(ckpt, export_dir, prompt: str = PROMPT) -> dict[str, dict]:
    export_dir = Path(export_dir)
    model, cfg = load_model(ckpt)
    tk = load_tokenizer(ckpt)
    ids = tk.encode(prompt, bos=True)
    with torch.no_grad():
        ref = model(torch.tensor([ids]))[0][0].numpy()
    res: dict[str, dict] = {}

    hf = export_dir / "hf"
    if hf.exists():
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            m = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.float32)
            t = AutoTokenizer.from_pretrained(hf)
            hf_ids = t(prompt)["input_ids"]
            assert hf_ids == ids, f"hf tokenizer mismatch {hf_ids[:8]} vs {ids[:8]}"
            with torch.no_grad():
                res["hf (transformers)"] = _cmp(ref, m(torch.tensor([ids])).logits[0].numpy())
        except ImportError:
            res["hf (transformers)"] = {"skipped": "transformers not installed"}

    for g in sorted((export_dir / "gguf").glob("*.gguf")) if (export_dir / "gguf").exists() else []:
        try:
            from llama_cpp import Llama

            llm = Llama(model_path=str(g), logits_all=True, n_ctx=cfg.max_seq_len, verbose=False, n_threads=4)
            toks = llm.tokenize(prompt.encode(), add_bos=True, special=False)
            assert toks == ids, f"gguf tokenizer mismatch {toks[:8]} vs {ids[:8]}"
            llm.reset()
            llm.eval(toks)
            got = np.array(llm.scores[: len(toks)])
            res[f"gguf {g.stem.split('-')[-1]} (llama.cpp)"] = _cmp(ref, got)
            del llm
        except ImportError:
            res[f"gguf {g.name}"] = {"skipped": "llama-cpp-python not installed"}

    onnx = next(iter(sorted((export_dir / "onnx").glob("*.onnx"))), None) if (export_dir / "onnx").exists() else None
    if onnx:
        try:
            import onnxruntime as ort

            sess = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
            got = sess.run(None, {"input_ids": np.array([ids], dtype=np.int64)})[0][0]
            res["onnx (onnxruntime)"] = _cmp(ref, got)
        except ImportError:
            res["onnx"] = {"skipped": "onnxruntime not installed"}

    npz = next(iter(sorted((export_dir / "numpy").glob("*.npz"))), None) if (export_dir / "numpy").exists() else None
    if npz:
        import importlib.util

        spec = importlib.util.spec_from_file_location("shellm_numpy", export_dir / "numpy" / "shellm_numpy.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        m = mod.Model(str(npz))
        assert m.tok.encode(prompt) == ids[1:], "numpy bpe mismatch"
        res["npz (pure numpy)"] = _cmp(ref, m.logits(ids))
    return res

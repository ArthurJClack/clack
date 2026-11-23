"""onnx: input_ids [batch, seq] -> logits [batch, seq, vocab].

runs in onnxruntime from python, c#, java, js (onnxruntime-web in a browser),
c++, rust... a tiny full-context generation loop lives in `onnx_generate`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..config import ModelConfig
from ..model import Shellm


class _LogitsOnly(torch.nn.Module):
    def __init__(self, m: Shellm):
        super().__init__()
        self.m = m

    def forward(self, input_ids):
        return self.m(input_ids)[0]


def export_onnx(state: dict, cfg: ModelConfig, out_path, opset: int = 18) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m = Shellm(cfg)
    m.load_state_dict(state, strict=False)
    w = _LogitsOnly(m.eval()).eval()
    ex = torch.randint(0, cfg.vocab_size, (1, 16))
    try:
        from torch.export import Dim

        seq = Dim("seq", min=1, max=cfg.max_seq_len)
        batch = Dim("batch", min=1, max=1024)
        prog = torch.onnx.export(
            w, (ex,), dynamo=True, opset_version=opset, input_names=["input_ids"], output_names=["logits"],
            dynamic_shapes={"input_ids": {0: batch, 1: seq}}, external_data=False,
        )
        prog.save(str(out_path))
    except Exception as e:  # older torch / exporter hiccup -> legacy exporter
        print(f"[onnx] dynamo exporter failed ({type(e).__name__}: {str(e)[:120]}), using legacy exporter")
        torch.onnx.export(
            w, (ex,), str(out_path), dynamo=False, opset_version=17, input_names=["input_ids"], output_names=["logits"],
            dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "logits": {0: "batch", 1: "seq"}},
        )
    return out_path


def onnx_generate(onnx_path, tokenizer, prompt: str, max_tokens: int = 60, temperature: float = 0.8, seed: int = 0, max_len: int = 256) -> str:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(seed)
    ids = tokenizer.encode(prompt, bos=True)
    out = []
    for _ in range(max_tokens):
        ctx = (ids + out)[-max_len:]
        logits = sess.run(None, {"input_ids": np.array([ctx], dtype=np.int64)})[0][0, -1].astype(np.float64)
        if temperature <= 0:
            t = int(logits.argmax())
        else:
            p = np.exp((logits - logits.max()) / temperature)
            p /= p.sum()
            t = int(rng.choice(len(p), p=p))
        if t == tokenizer.eos_id:
            break
        out.append(t)
    return tokenizer.decode(out)

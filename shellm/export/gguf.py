"""gguf: the llama.cpp file format (also what ollama, lm studio, jan, koboldcpp
and gpt4all load).

we write the "llama" architecture with a gpt-2 style byte-level bpe vocab, so
stock llama.cpp runs it with zero patches. quant types writable from python:
f32, f16, bf16, q8_0, q5_1, q5_0, q4_1, q4_0. for k-quants (q4_k_m etc) run
llama.cpp's `llama-quantize` on the f16 file.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import ModelConfig

QTYPES = {
    "f32": ("F32", "ALL_F32"),
    "f16": ("F16", "MOSTLY_F16"),
    "bf16": ("BF16", "MOSTLY_BF16"),
    "q8_0": ("Q8_0", "MOSTLY_Q8_0"),
    "q5_1": ("Q5_1", "MOSTLY_Q5_1"),
    "q5_0": ("Q5_0", "MOSTLY_Q5_0"),
    "q4_1": ("Q4_1", "MOSTLY_Q4_1"),
    "q4_0": ("Q4_0", "MOSTLY_Q4_0"),
}


def permute_rope(w: np.ndarray, n_head: int) -> np.ndarray:
    """hf rotate_half layout -> llama.cpp interleaved rope layout (same as convert_hf_to_gguf)."""
    return w.reshape(n_head, 2, w.shape[0] // n_head // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


def gguf_tensors(state: dict, cfg: ModelConfig) -> dict[str, np.ndarray]:
    def g(k):
        return state[k].detach().float().cpu().numpy()

    out = {"token_embd.weight": g("model.embed_tokens.weight"), "output_norm.weight": g("model.norm.weight")}
    if not cfg.tie_embeddings:
        out["output.weight"] = g("lm_head.weight")
    for i in range(cfg.n_layers):
        p = f"model.layers.{i}."
        b = f"blk.{i}."
        out[b + "attn_norm.weight"] = g(p + "input_layernorm.weight")
        out[b + "attn_q.weight"] = permute_rope(g(p + "self_attn.q_proj.weight"), cfg.n_heads)
        out[b + "attn_k.weight"] = permute_rope(g(p + "self_attn.k_proj.weight"), cfg.n_kv_heads)
        out[b + "attn_v.weight"] = g(p + "self_attn.v_proj.weight")
        out[b + "attn_output.weight"] = g(p + "self_attn.o_proj.weight")
        out[b + "ffn_norm.weight"] = g(p + "post_attention_layernorm.weight")
        out[b + "ffn_gate.weight"] = g(p + "mlp.gate_proj.weight")
        out[b + "ffn_up.weight"] = g(p + "mlp.up_proj.weight")
        out[b + "ffn_down.weight"] = g(p + "mlp.down_proj.weight")
    return out


def export_gguf(state: dict, cfg: ModelConfig, tokenizer, out_path, qtype: str = "f16", name: str = "shellm", info: dict | None = None) -> Path:
    import gguf
    from gguf import quants

    qname, ftype = QTYPES[qtype]
    qt = getattr(gguf.GGMLQuantizationType, qname)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    w = gguf.GGUFWriter(str(out_path), "llama")
    w.add_name(name)
    w.add_description("shellm: a personal llama-compatible model built from scratch")
    w.add_context_length(cfg.max_seq_len)
    w.add_embedding_length(cfg.dim)
    w.add_block_count(cfg.n_layers)
    w.add_feed_forward_length(cfg.ffn_dim)
    w.add_rope_dimension_count(cfg.head_dim)
    w.add_head_count(cfg.n_heads)
    w.add_head_count_kv(cfg.n_kv_heads)
    w.add_layer_norm_rms_eps(cfg.norm_eps)
    w.add_rope_freq_base(cfg.rope_theta)
    w.add_vocab_size(cfg.vocab_size)
    w.add_file_type(getattr(gguf.LlamaFileType, ftype))

    tokens, merges = tokenizer.vocab_and_merges()
    specials = {cfg.pad_id, cfg.bos_id, cfg.eos_id}
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("gpt-2")
    w.add_token_list(tokens)
    w.add_token_types([gguf.TokenType.CONTROL if i in specials else gguf.TokenType.NORMAL for i in range(len(tokens))])
    w.add_token_merges(merges)
    w.add_bos_token_id(cfg.bos_id)
    w.add_eos_token_id(cfg.eos_id)
    w.add_pad_token_id(cfg.pad_id)
    w.add_add_bos_token(True)
    w.add_add_eos_token(False)
    if info:
        w.add_string("shellm.made_by", str(info.get("made_by", "")))
        if info.get("recipe"):
            w.add_string("shellm.recipe", str(info["recipe"]))

    for tname, arr in gguf_tensors(state, cfg).items():
        if arr.ndim == 1 or qt == gguf.GGMLQuantizationType.F32:
            w.add_tensor(tname, arr.astype(np.float32))
        elif qt == gguf.GGMLQuantizationType.F16:
            w.add_tensor(tname, arr.astype(np.float16))
        else:
            try:
                w.add_tensor(tname, quants.quantize(arr, qt), raw_dtype=qt)
            except quants.QuantError:  # row not a multiple of the block size
                w.add_tensor(tname, arr.astype(np.float16))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return out_path


MODELFILE = """# ollama: `ollama create {name} -f Modelfile` then `ollama run {name}`
FROM ./{gguf}
TEMPLATE \"\"\"{{{{ .Prompt }}}}\"\"\"
PARAMETER temperature 0.8
PARAMETER top_k 50
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx {ctx}
PARAMETER stop "</s>"
"""


def write_modelfile(out_dir, gguf_name: str, name: str, ctx: int) -> Path:
    p = Path(out_dir) / "Modelfile"
    p.write_text(MODELFILE.format(name=name, gguf=gguf_name, ctx=ctx))
    return p

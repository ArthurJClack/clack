"""huggingface format: config.json + model.safetensors + tokenizer files.

loads with `transformers.AutoModelForCausalLM.from_pretrained(dir)` and from
there into anything that eats hf llama checkpoints (mlx-lm, vllm, llama.cpp's
convert_hf_to_gguf.py, candle, ...).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from ..config import ModelConfig


def hf_config(cfg: ModelConfig, dtype: str = "float32") -> dict:
    return {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.dim,
        "intermediate_size": cfg.ffn_dim,
        "num_hidden_layers": cfg.n_layers,
        "num_attention_heads": cfg.n_heads,
        "num_key_value_heads": cfg.n_kv_heads,
        "head_dim": cfg.head_dim,
        "hidden_act": "silu",
        "max_position_embeddings": cfg.max_seq_len,
        "rms_norm_eps": cfg.norm_eps,
        "rope_theta": cfg.rope_theta,
        "tie_word_embeddings": cfg.tie_embeddings,
        "attention_bias": False,
        "mlp_bias": False,
        "bos_token_id": cfg.bos_id,
        "eos_token_id": cfg.eos_id,
        "pad_token_id": cfg.pad_id,
        "torch_dtype": dtype,
        "dtype": dtype,
    }


def export_hf(state: dict, cfg: ModelConfig, tokenizer_json: str | Path, out_dir, dtype=torch.float16) -> Path:
    from safetensors.torch import save_file

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().to(dtype).cpu().contiguous() for k, v in state.items()}
    if cfg.tie_embeddings:
        sd.pop("lm_head.weight", None)
    save_file(sd, str(out / "model.safetensors"), metadata={"format": "pt"})
    dname = str(dtype).removeprefix("torch.")
    (out / "config.json").write_text(json.dumps(hf_config(cfg, dname), indent=2))
    (out / "generation_config.json").write_text(
        json.dumps(
            {"bos_token_id": cfg.bos_id, "eos_token_id": cfg.eos_id, "pad_token_id": cfg.pad_id,
             "do_sample": True, "temperature": 0.8, "top_k": 50, "top_p": 0.95, "repetition_penalty": 1.1},
            indent=2,
        )
    )
    # bake "prepend <s>" into the tokenizer so hf/transformers.js users get it for free
    from tokenizers import Tokenizer as HFTok
    from tokenizers.processors import TemplateProcessing

    tk = HFTok.from_file(str(tokenizer_json))
    tk.post_processor = TemplateProcessing(
        single="<s> $A", pair="<s> $A <s> $B", special_tokens=[("<s>", cfg.bos_id)]
    )
    tk.save(str(out / "tokenizer.json"))
    (out / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "pad_token": "<pad>",
                "add_bos_token": True,
                "add_eos_token": False,
                "model_max_length": cfg.max_seq_len,
                "clean_up_tokenization_spaces": False,
            },
            indent=2,
        )
    )
    (out / "special_tokens_map.json").write_text(
        json.dumps({"bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>"}, indent=2)
    )
    return out

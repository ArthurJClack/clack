"""lora: train a low-rank delta instead of the full weights.

a lora adapter *is* a weight-space object: delta_W = (alpha/r) * B @ A.
adapters are saved in peft format (adapter_model.safetensors +
adapter_config.json) so huggingface peft and llama.cpp's convert_lora_to_gguf
can read them too.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn

DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        self.r, self.alpha = r, alpha
        self.scale = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        base.weight.requires_grad_(False)

    @property
    def weight(self):  # so code that reads .weight still sees the effective matrix
        return self.base.weight + self.delta()

    def delta(self):
        return (self.lora_B @ self.lora_A) * self.scale

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.t() @ self.lora_B.t()) * self.scale


def apply_lora(model: nn.Module, r: int = 8, alpha: float = 16.0, targets=DEFAULT_TARGETS) -> list[str]:
    for p in model.parameters():
        p.requires_grad_(False)
    wrapped = []
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            if isinstance(child, nn.Linear) and child_name in targets:
                setattr(mod, child_name, LoRALinear(child, r, alpha))
                wrapped.append(f"{name}.{child_name}" if name else child_name)
    return wrapped


def lora_modules(model: nn.Module):
    return {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}


def lora_deltas(model: nn.Module) -> dict[str, torch.Tensor]:
    """{param_name: delta_W} in plain model naming - a task vector."""
    return {f"{n}.weight": m.delta().detach().clone() for n, m in lora_modules(model).items()}


def unwrap_lora(model: nn.Module, merge: bool = True) -> nn.Module:
    """replace LoRALinear with plain Linear (optionally folding the delta in)."""
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            if isinstance(child, LoRALinear):
                base = child.base
                if merge:
                    with torch.no_grad():
                        base.weight += child.delta()
                base.weight.requires_grad_(True)
                setattr(mod, child_name, base)
    return model


def save_adapter(model: nn.Module, out_dir, base_ckpt: str = "") -> Path:
    from safetensors.torch import save_file

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mods = lora_modules(model)
    if not mods:
        raise ValueError("model has no lora modules")
    tensors, r, alpha, targets = {}, None, None, set()
    for n, m in mods.items():
        tensors[f"base_model.model.{n}.lora_A.weight"] = m.lora_A.detach().cpu().contiguous()
        tensors[f"base_model.model.{n}.lora_B.weight"] = m.lora_B.detach().cpu().contiguous()
        r, alpha = m.r, m.alpha
        targets.add(n.rsplit(".", 1)[-1])
    save_file(tensors, str(out / "adapter_model.safetensors"))
    (out / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": r,
                "lora_alpha": alpha,
                "lora_dropout": 0.0,
                "bias": "none",
                "target_modules": sorted(targets),
                "base_model_name_or_path": str(base_ckpt),
                "fan_in_fan_out": False,
                "inference_mode": True,
            },
            indent=2,
        )
    )
    return out


def load_adapter_deltas(adapter_dir) -> dict[str, torch.Tensor]:
    """peft adapter dir -> {param_name: delta_W}."""
    from safetensors.torch import load_file

    d = Path(adapter_dir)
    cfg = json.loads((d / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    t = load_file(str(d / "adapter_model.safetensors"))
    out = {}
    for k, A in t.items():
        if ".lora_A." not in k:
            continue
        B = t[k.replace(".lora_A.", ".lora_B.")]
        name = k.removeprefix("base_model.model.").replace(".lora_A.weight", ".weight")
        out[name] = (B.float() @ A.float()) * scale
    return out

"""pretraining + finetuning (full or lora) loop."""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .config import ModelConfig, preset
from .data import TokenData
from .io import load_model, pick_device, save_checkpoint
from .lora import apply_lora, save_adapter, unwrap_lora
from .model import Shellm


@dataclass
class TrainArgs:
    data: str
    out: str
    preset: str = "nano"
    init_from: str | None = None  # checkpoint dir -> finetune
    tokenizer: str | None = None
    steps: int = 2000
    batch_size: int = 32
    seq_len: int = 256
    grad_accum: int = 1
    lr: float = 1e-3
    min_lr_frac: float = 0.1
    warmup: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 250
    eval_batches: int = 20
    log_every: int = 25
    device: str = "auto"
    dtype: str = "float32"  # bfloat16 on gpus that support it
    compile: bool = False
    threads: int | None = None
    seed: int = 1337
    lora_r: int = 0  # >0 -> train a lora adapter instead of full weights
    lora_alpha: float = 16.0
    merge_lora: bool = True  # also write the merged full checkpoint
    time_limit_min: float | None = None
    overrides: dict = field(default_factory=dict)


@torch.no_grad()
def evaluate(model, data: TokenData, batch_size: int, seq_len: int, n_batches: int, device, ctx) -> float:
    model.eval()
    losses = []
    for x, y in data.sequential(batch_size, seq_len, n_batches):
        with ctx:
            _, loss = model(x.to(device), y.to(device))
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def lr_at(step: int, a: TrainArgs) -> float:
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    prog = min(1.0, (step - a.warmup) / max(1, a.steps - a.warmup))
    return a.lr * (a.min_lr_frac + (1 - a.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * prog)))


def train(a: TrainArgs) -> Path:
    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = pick_device(a.device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    train_data = TokenData(a.data, "train")
    val_data = TokenData(a.data, "val")
    tok_path = a.tokenizer or train_data.meta["tokenizer"]

    if a.init_from:
        model, cfg = load_model(a.init_from, device)
        tok_path = a.tokenizer or str(Path(a.init_from) / "tokenizer.json")
    else:
        from .tokenizer import Tokenizer

        tk = Tokenizer(tok_path)
        cfg = preset(a.preset, vocab_size=tk.vocab_size, bos_id=tk.bos_id, eos_id=tk.eos_id, pad_id=tk.pad_id, **a.overrides)
        model = Shellm(cfg).to(device)
    if a.seq_len > cfg.max_seq_len:
        raise SystemExit(f"--seq-len {a.seq_len} > model max_seq_len {cfg.max_seq_len}")

    lora_names = []
    if a.lora_r > 0:
        lora_names = apply_lora(model, a.lora_r, a.lora_alpha)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.dim() >= 2]
    no_decay = [p for p in params if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": a.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=a.lr,
        betas=(0.9, 0.95),
        fused=device == "cuda",
    )
    pt_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[a.dtype]
    use_amp = pt_dtype != torch.float32
    ctx = torch.autocast(device_type=device.split(":")[0], dtype=pt_dtype) if use_amp else torch.autocast("cpu", enabled=False)
    scaler = torch.amp.GradScaler(enabled=pt_dtype == torch.float16)
    step_fn = torch.compile(model) if a.compile else model

    n_train = sum(p.numel() for p in params)
    print(f"[shellm] device={device} params={model.n_params()/1e6:.2f}M trainable={n_train/1e6:.2f}M "
          f"train_tokens={len(train_data)/1e6:.1f}M lora={'r=%d' % a.lora_r if a.lora_r else 'off'}")
    (out / "train_args.json").write_text(json.dumps(a.__dict__, indent=2, default=str))
    log = open(out / "log.jsonl", "a")

    best = float("inf")
    t0 = time.time()
    tokens_per_step = a.batch_size * a.seq_len * a.grad_accum
    info = {"made_by": "train", "args": a.__dict__}

    def save(tag: str, val_loss: float):
        nonlocal info
        info = {**info, "step": step, "val_loss": val_loss, "tokens_seen": step * tokens_per_step}
        if lora_names:
            save_adapter(model, out / f"adapter-{tag}", base_ckpt=a.init_from or "")
        else:
            save_checkpoint(out / tag, model.state_dict(), cfg, tok_path, info)

    step = 0
    for step in range(a.steps + 1):
        if step % a.eval_every == 0 or step == a.steps:
            vl = evaluate(model, val_data, a.batch_size, a.seq_len, a.eval_batches, device, ctx)
            rec = {"step": step, "val_loss": vl, "val_ppl": math.exp(min(vl, 20)), "elapsed_s": time.time() - t0}
            print(f"[eval] step {step:5d} val_loss {vl:.4f} ppl {rec['val_ppl']:.1f}")
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if vl < best:
                best = vl
                save("best", vl)
        if step == a.steps:
            break
        if a.time_limit_min and (time.time() - t0) / 60 > a.time_limit_min:
            print(f"[shellm] time limit hit at step {step}")
            break

        for g in opt.param_groups:
            g["lr"] = lr_at(step, a)
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(a.grad_accum):
            x, y = train_data.batch(a.batch_size, a.seq_len, rng)
            with ctx:
                _, loss = step_fn(x.to(device), y.to(device))
            scaler.scale(loss / a.grad_accum).backward()
            tot += loss.item() / a.grad_accum
        if a.grad_clip:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, a.grad_clip)
        scaler.step(opt)
        scaler.update()

        if step % a.log_every == 0:
            el = time.time() - t0
            tps = (step + 1) * tokens_per_step / max(el, 1e-9)
            print(f"step {step:5d} loss {tot:.4f} lr {lr_at(step, a):.2e} {tps:,.0f} tok/s {el/60:.1f}m")
            log.write(json.dumps({"step": step, "loss": tot, "tok_s": tps}) + "\n")

    vl = evaluate(model, val_data, a.batch_size, a.seq_len, a.eval_batches, device, ctx)
    save("last", vl)
    final = out / "last"
    if lora_names and a.merge_lora:
        unwrap_lora(model, merge=True)
        info = {**info, "made_by": "lora-merge", "lora_r": a.lora_r}
        save_checkpoint(out / "merged", model.state_dict(), cfg, tok_path, info)
        final = out / "merged"
    log.close()
    print(f"[shellm] done. best val_loss {best:.4f}. wrote {out}")
    return final

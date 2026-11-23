"""shellm command line.

    python -m shellm tokenizer --src corpus --vocab 8192 --out runs/tok.json
    python -m shellm data      --src corpus --tokenizer runs/tok.json --out runs/data/books
    python -m shellm train     --data runs/data/books --out runs/base --preset nano --steps 3000
    python -m shellm sample    runs/base/best "Once upon a time"
    python -m shellm eval      runs/base/best --data runs/data/books
    python -m shellm export    runs/base/best --out dist/shellm --formats hf,gguf,onnx,npz
    python -m shellm serve     runs/base/best          # api + live 3d viewer on :8000
    python -m shellm wsp       recipes/01_task_vectors.py
    python -m shellm info      runs/base/best
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _list(s: str | None) -> list[str] | None:
    return [x for x in s.split(",") if x] if s else None


def cmd_tokenizer(a):
    from .data import iter_texts
    from .tokenizer import train_tokenizer

    tk = train_tokenizer(iter_texts(a.src, _list(a.include), _list(a.exclude), int(a.max_mb * 1e6)), a.vocab, a.out)
    sample = "It is a truth universally acknowledged, that a single man in possession of a good fortune"
    ids = tk.encode(sample)
    print(f"vocab {tk.vocab_size}; {len(sample)} chars -> {len(ids)} tokens; roundtrip ok: {tk.decode(ids) == sample}")


def cmd_data(a):
    from .data import prepare
    from .tokenizer import Tokenizer

    meta = prepare(a.src, a.out, Tokenizer(a.tokenizer), a.val_frac, _list(a.include), _list(a.exclude))
    print(json.dumps(meta, indent=2))


def cmd_train(a):
    from .train import TrainArgs, train

    keys = TrainArgs.__dataclass_fields__.keys()
    kw = {k: v for k, v in vars(a).items() if k in keys and v is not None}
    train(TrainArgs(**kw))


def cmd_sample(a):
    import torch

    from .io import load_model, load_tokenizer, pick_device

    dev = pick_device(a.device)
    model, cfg = load_model(a.ckpt, dev)
    tk = load_tokenizer(a.ckpt)
    if a.seed is not None:
        torch.manual_seed(a.seed)
    for i in range(a.n):
        ids = torch.tensor([tk.encode(a.prompt, bos=True)], device=dev)
        print(a.prompt, end="", flush=True)
        buf: list[int] = []
        printed = [0]

        def stream(t):
            buf.append(t)
            text = tk.decode(buf)
            if not text.endswith("�"):  # wait for complete utf-8
                print(text[printed[0]:], end="", flush=True)
                printed[0] = len(text)

        model.generate(ids, a.max_tokens, a.temperature, a.top_k, a.top_p, a.rep_penalty, eos_id=tk.eos_id, stream=stream)
        print("\n" + ("-" * 40 if i < a.n - 1 else ""))


def cmd_eval(a):
    from .evaluate import eval_ckpt

    for c in a.ckpts:
        r = eval_ckpt(c, a.data, a.batches, a.seq_len)
        print(f"{c}: " + "  ".join(f"{k}={v:.4f}" for k, v in r.items()))


def cmd_export(a):
    from .export import export_all

    for fmt, path in export_all(a.ckpt, a.out, a.formats.split(","), a.name, a.quant.split(",")).items():
        print(f"  {fmt:10s} -> {path}")
    if a.verify:
        from .export.verify import verify_exports

        print("verifying against the torch model:")
        for fmt, r in verify_exports(a.ckpt, a.out).items():
            print(f"  {fmt:28s} " + "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))


def cmd_wsp(a):
    from .wsp.recipe import run_recipe

    run_recipe(a.recipe, a.args)


def cmd_serve(a):
    from .server import serve

    serve(a.ckpt, a.host, a.port, a.device, a.data, a.cors)


def cmd_info(a):
    from .io import load_state

    sd, cfg = load_state(a.ckpt)
    n = sum(v.numel() for v in sd.values())
    print(json.dumps(cfg.to_dict(), indent=2))
    print(f"params: {n:,} ({n/1e6:.2f}M)  tensors: {len(sd)}")
    info = Path(a.ckpt) / "info.json"
    if info.exists():
        i = json.loads(info.read_text())
        print("made_by:", i.get("made_by"), "| val_loss:", i.get("val_loss"), "| recipe:", i.get("recipe", "-"))


def main(argv=None):
    p = argparse.ArgumentParser(prog="shellm", description="your own tiny llm + weight-space programming")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("tokenizer", help="train a byte-level bpe tokenizer")
    s.add_argument("--src", action="append", required=True)
    s.add_argument("--vocab", type=int, default=8192)
    s.add_argument("--out", required=True)
    s.add_argument("--max-mb", type=float, default=60)
    s.add_argument("--include")
    s.add_argument("--exclude")
    s.set_defaults(fn=cmd_tokenizer)

    s = sub.add_parser("data", help="tokenize a corpus into train/val .bin")
    s.add_argument("--src", action="append", required=True, help="dir or file, optional :weight suffix")
    s.add_argument("--tokenizer", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--val-frac", type=float, default=0.01)
    s.add_argument("--include", help="comma list of globs (relative paths)")
    s.add_argument("--exclude", help="comma list of globs")
    s.set_defaults(fn=cmd_data)

    s = sub.add_parser("train", help="pretrain or finetune")
    s.add_argument("--data", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--preset")
    s.add_argument("--init-from", dest="init_from")
    s.add_argument("--tokenizer")
    s.add_argument("--steps", type=int)
    s.add_argument("--batch-size", dest="batch_size", type=int)
    s.add_argument("--seq-len", dest="seq_len", type=int)
    s.add_argument("--grad-accum", dest="grad_accum", type=int)
    s.add_argument("--lr", type=float)
    s.add_argument("--warmup", type=int)
    s.add_argument("--weight-decay", dest="weight_decay", type=float)
    s.add_argument("--eval-every", dest="eval_every", type=int)
    s.add_argument("--eval-batches", dest="eval_batches", type=int)
    s.add_argument("--device")
    s.add_argument("--dtype")
    s.add_argument("--compile", action="store_true", default=None)
    s.add_argument("--threads", type=int)
    s.add_argument("--seed", type=int)
    s.add_argument("--lora-r", dest="lora_r", type=int)
    s.add_argument("--lora-alpha", dest="lora_alpha", type=float)
    s.add_argument("--time-limit-min", dest="time_limit_min", type=float)
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("sample", help="generate text")
    s.add_argument("ckpt")
    s.add_argument("prompt", nargs="?", default="")
    s.add_argument("--max-tokens", type=int, default=200)
    s.add_argument("--temperature", type=float, default=0.8)
    s.add_argument("--top-k", type=int, default=50)
    s.add_argument("--top-p", type=float, default=0.95)
    s.add_argument("--rep-penalty", type=float, default=1.1)
    s.add_argument("-n", type=int, default=1)
    s.add_argument("--seed", type=int)
    s.add_argument("--device", default="auto")
    s.set_defaults(fn=cmd_sample)

    s = sub.add_parser("eval", help="validation loss / perplexity")
    s.add_argument("ckpts", nargs="+")
    s.add_argument("--data", action="append", required=True, help="token data dir(s)")
    s.add_argument("--batches", type=int, default=40)
    s.add_argument("--seq-len", type=int, default=256)
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("export", help="write the model in other formats")
    s.add_argument("ckpt")
    s.add_argument("--out", required=True)
    s.add_argument("--formats", default="hf,gguf,onnx,npz")
    s.add_argument("--quant", default="f16,q8_0", help="gguf types: f32,f16,q8_0")
    s.add_argument("--name", default="shellm")
    s.add_argument("--verify", action="store_true", help="check every format against torch logits")
    s.set_defaults(fn=cmd_export)

    s = sub.add_parser("wsp", help="run a weight-space program (recipe .py)")
    s.add_argument("recipe")
    s.add_argument("args", nargs=argparse.REMAINDER)
    s.set_defaults(fn=cmd_wsp)

    s = sub.add_parser("serve", help="http api (openai-compatible) + live 3d viewer")
    s.add_argument("ckpt")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--device", default="auto")
    s.add_argument("--data", help="token data dir: better statistics for live fact edits")
    s.add_argument("--cors", action="store_true", help="allow cross-origin browser requests")
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("info", help="describe a checkpoint")
    s.add_argument("ckpt")
    s.set_defaults(fn=cmd_info)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

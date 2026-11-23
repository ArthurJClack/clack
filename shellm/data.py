"""corpus -> token .bin files.

sources are directories (or files) of .txt / .md. add your own writing, notes,
chat logs, whatever into data/personal/ and include it with a weight, e.g.

    shellm data --src corpus/books --src data/personal:20 --out runs/data/mix

a weight of 20 repeats that source 20x so it isn't drowned out.
"""
from __future__ import annotations

import fnmatch
import json
import random
import re
from pathlib import Path
from typing import Iterator

import numpy as np

TEXT_EXT = {".txt", ".md", ".markdown", ".text"}


def parse_src(spec: str) -> tuple[Path, int]:
    path, _, w = spec.rpartition(":") if re.search(r":\d+$", spec) else (spec, "", "1")
    return Path(path), int(w)


def iter_files(src: Path, include: list[str] | None = None, exclude: list[str] | None = None) -> Iterator[Path]:
    files = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.suffix.lower() in TEXT_EXT)
    for f in files:
        rel = str(f.relative_to(src)) if src.is_dir() else f.name
        if include and not any(fnmatch.fnmatch(rel, pat) for pat in include):
            continue
        if exclude and any(fnmatch.fnmatch(rel, pat) for pat in exclude):
            continue
        if f.name.lower() in {"readme.md", "license.md", "license.txt"}:
            continue
        yield f


_md_head = re.compile(r"^#{1,6}\s*", re.M)
_md_meta = re.compile(r"^#*\s*(Title|Author|Year):.*$", re.M)
_rule = re.compile(r"^\s*([-*_=]\s*){3,}$", re.M)
_ital = re.compile(r"(?<!\w)[_*]{1,2}([^_*\n]+?)[_*]{1,2}(?!\w)")
_ws = re.compile(r"[ \t]+")
_nl = re.compile(r"\n{3,}")


def clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _md_meta.sub("", text)
    text = _rule.sub("", text)
    text = _md_head.sub("", text)
    text = _ital.sub(r"\1", text)
    text = _ws.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _nl.sub("\n\n", text)
    return text.strip()


def chunks(text: str, size: int = 32_000) -> list[str]:
    """split on paragraph boundaries into ~size-char pieces (separators kept)."""
    out, cur, n = [], [], 0
    for para in text.split("\n\n"):
        cur.append(para + "\n\n")
        n += len(para) + 2
        if n >= size:
            out.append("".join(cur))
            cur, n = [], 0
    if cur:
        out.append("".join(cur))
    if out:
        out[-1] = out[-1].rstrip("\n")
    return out


def iter_texts(srcs: list[str], include=None, exclude=None, max_chars: int | None = None) -> Iterator[str]:
    """plain text stream (for tokenizer training)."""
    total = 0
    for spec in srcs:
        src, _ = parse_src(spec)
        for f in iter_files(src, include, exclude):
            for c in chunks(clean(f.read_text(errors="ignore"))):
                yield c
                total += len(c)
                if max_chars and total >= max_chars:
                    return


def prepare(
    srcs: list[str],
    out_dir: str | Path,
    tokenizer,
    val_frac: float = 0.01,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    seed: int = 0,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    train_docs: list[np.ndarray] = []
    val_docs: list[np.ndarray] = []
    files_used = []
    for spec in srcs:
        src, weight = parse_src(spec)
        for f in iter_files(src, include, exclude):
            text = clean(f.read_text(errors="ignore"))
            if len(text) < 200:
                continue
            pieces = chunks(text)
            enc = tokenizer.encode_batch(pieces)
            # hold out the tail of every document so val covers every source
            n_val = max(1, int(round(len(enc) * val_frac))) if len(enc) > 4 else 0
            dt = np.uint16 if tokenizer.vocab_size < 65536 else np.uint32
            bos, eos = np.array([tokenizer.bos_id], dt), np.array([tokenizer.eos_id], dt)
            tr = [np.asarray(e, dtype=dt) for e in enc[: len(enc) - n_val]]
            va = [np.asarray(e, dtype=dt) for e in enc[len(enc) - n_val :]]
            doc = np.concatenate([bos, *tr, eos])
            train_docs.extend([doc] * weight)
            if va:
                val_docs.append(np.concatenate([bos, *va, eos]))
            files_used.append(str(f))
    if not train_docs:
        raise SystemExit("no documents found - check --src / --include / --exclude")
    rng.shuffle(train_docs)
    dt = np.uint16 if tokenizer.vocab_size < 65536 else np.uint32
    train = np.concatenate(train_docs).astype(dt)
    val = np.concatenate(val_docs).astype(dt) if val_docs else train[-10_000:]
    train.tofile(out_dir / "train.bin")
    val.tofile(out_dir / "val.bin")
    meta = {
        "dtype": np.dtype(dt).name,
        "vocab_size": tokenizer.vocab_size,
        "tokenizer": str(Path(tokenizer.path).resolve()),
        "train_tokens": int(train.size),
        "val_tokens": int(val.size),
        "files": len(files_used),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    (out_dir / "files.txt").write_text("\n".join(files_used))
    return meta


class TokenData:
    def __init__(self, data_dir: str | Path, split: str = "train"):
        self.dir = Path(data_dir)
        meta = json.loads((self.dir / "meta.json").read_text())
        self.meta = meta
        self.arr = np.memmap(self.dir / f"{split}.bin", dtype=np.dtype(meta["dtype"]), mode="r")

    def __len__(self):
        return len(self.arr)

    def batch(self, B: int, T: int, rng: np.random.Generator):
        import torch

        ix = rng.integers(0, len(self.arr) - T - 1, size=B)
        x = np.stack([self.arr[i : i + T].astype(np.int64) for i in ix])
        y = np.stack([self.arr[i + 1 : i + 1 + T].astype(np.int64) for i in ix])
        return torch.from_numpy(x), torch.from_numpy(y)

    def sequential(self, B: int, T: int, max_batches: int | None = None):
        """deterministic, non-overlapping windows (for eval)."""
        import torch

        n = (len(self.arr) - 1) // T
        starts = np.arange(n) * T
        for bi in range(0, n, B):
            if max_batches is not None and bi // B >= max_batches:
                return
            s = starts[bi : bi + B]
            x = np.stack([self.arr[i : i + T].astype(np.int64) for i in s])
            y = np.stack([self.arr[i + 1 : i + 1 + T].astype(np.int64) for i in s])
            yield torch.from_numpy(x), torch.from_numpy(y)

"""byte-level bpe tokenizer (gpt-2 style), trained on your own corpus.

byte-level means it can encode *any* text (no unknown tokens), and the gpt-2
pre-tokenizer regex is understood natively by llama.cpp ("gpt2"/"gpt-2"), so the
same vocab + merges go straight into gguf.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

SPECIALS = ["<pad>", "<s>", "</s>"]  # ids 0, 1, 2


class Tokenizer:
    def __init__(self, path: str | Path):
        from tokenizers import Tokenizer as HFTok

        self.path = Path(path)
        self.tk = HFTok.from_file(str(self.path))
        self.pad_id = self.tk.token_to_id("<pad>")
        self.bos_id = self.tk.token_to_id("<s>")
        self.eos_id = self.tk.token_to_id("</s>")

    @property
    def vocab_size(self) -> int:
        return self.tk.get_vocab_size()

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> list[int]:
        ids = self.tk.encode(text, add_special_tokens=False).ids
        return ([self.bos_id] if bos else []) + ids + ([self.eos_id] if eos else [])

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [e.ids for e in self.tk.encode_batch(texts, add_special_tokens=False)]

    def decode(self, ids: Iterable[int]) -> str:
        return self.tk.decode(list(ids), skip_special_tokens=True)

    def id_to_token(self, i: int) -> str:
        return self.tk.id_to_token(i)

    # vocab + merges in rank order, for gguf export
    def vocab_and_merges(self) -> tuple[list[str], list[str]]:
        model = json.loads(self.tk.to_str())["model"]
        vocab = sorted(model["vocab"].items(), key=lambda kv: kv[1])
        tokens = [t for t, _ in vocab]
        assert [i for _, i in vocab] == list(range(len(tokens))), "vocab ids must be dense"
        merges = [m if isinstance(m, str) else " ".join(m) for m in model["merges"]]
        return tokens, merges


def train_tokenizer(texts: Iterator[str], vocab_size: int, out: str | Path) -> Tokenizer:
    from tokenizers import Tokenizer as HFTok
    from tokenizers import decoders, models, pre_tokenizers, trainers

    tk = HFTok(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tk.train_from_iterator(texts, trainer=trainer)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tk.save(str(out))
    return Tokenizer(out)

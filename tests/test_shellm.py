"""fast end-to-end tests on a tiny model (run: python -m pytest tests -q)."""
from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from shellm.config import preset
from shellm.data import prepare
from shellm.io import load_model, save_checkpoint
from shellm.lora import apply_lora, lora_deltas, unwrap_lora
from shellm.model import Shellm
from shellm.tokenizer import train_tokenizer
from shellm.wsp import Fact, Weights, ablate_direction, fact_logprob, linear, rome, slerp, stack_layers, ties

WORDS = "the cat sat on a mat while dog ran far away from home and then came back to sleep in sun".split()


@pytest.fixture(scope="session")
def tiny(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny")
    rng = random.Random(0)
    corpus = d / "corpus"
    corpus.mkdir()
    for i in range(6):
        paras = [" ".join(rng.choice(WORDS) for _ in range(40)).capitalize() + "." for _ in range(80)]
        (corpus / f"doc{i}.txt").write_text("\n\n".join(paras))
    tk = train_tokenizer(((corpus / f"doc{i}.txt").read_text() for i in range(6)), 300, d / "tok.json")
    prepare([str(corpus)], d / "data", tk, val_frac=0.2)
    # a few seconds of training so logits are confident enough to be interesting
    from shellm.train import TrainArgs, train

    ck = train(TrainArgs(data=str(d / "data"), out=str(d / "run"), preset="test", steps=80, batch_size=16, seq_len=64,
                         lr=3e-3, warmup=10, eval_every=1000, log_every=1000, threads=2))
    cfg = load_model(ck)[1]
    return {"dir": d, "tk": tk, "cfg": cfg, "ckpt": ck, "data": d / "data"}


def test_kv_cache_matches_full_forward(tiny):
    m, cfg = load_model(tiny["ckpt"])
    x = torch.randint(0, cfg.vocab_size, (2, 30))
    full = m(x)[0]
    caches = [dict() for _ in m.model.layers]
    parts = [m(x[:, :10], caches=caches)[0], m(x[:, 10:11], caches=caches, start_pos=10)[0], m(x[:, 11:], caches=caches, start_pos=11)[0]]
    assert torch.allclose(full, torch.cat(parts, 1), atol=1e-5)


def test_lora_delta_equals_merge(tiny):
    m, _ = load_model(tiny["ckpt"])
    base = {k: v.clone() for k, v in m.state_dict().items()}
    apply_lora(m, r=4, alpha=8)
    for n, p in m.named_parameters():
        if "lora_B" in n:
            torch.nn.init.normal_(p, 0, 0.1)
    x = torch.randint(0, 100, (1, 12))
    with torch.no_grad():
        y_lora = m(x)[0]
    deltas = lora_deltas(m)
    unwrap_lora(m, merge=True)
    with torch.no_grad():
        assert torch.allclose(y_lora, m(x)[0], atol=1e-4)
    for k, d in deltas.items():
        assert torch.allclose(base[k] + d, m.state_dict()[k], atol=1e-6)


def test_weight_arithmetic_identities(tiny):
    a = Weights.load(tiny["ckpt"])
    b = a.map(lambda k, v: v + 0.01 * torch.randn_like(v))
    tv = b - a
    assert tv.kind == "delta"
    back = a + tv
    assert back.kind == "model"
    assert all(torch.allclose(back[k], b[k], atol=1e-6) for k in a.keys())
    assert abs(linear([a, b], [1.0, 0.0]).cosine(a) - 1) < 1e-6
    s = slerp(a, b, 0.0)
    assert all(torch.allclose(s[k], a[k], atol=1e-5) for k in a.keys())
    mlp = tv["*.mlp.*"]
    assert len(mlp) and all(".mlp." in k for k in mlp.keys())
    partial = a + mlp  # union semantics: only mlps move
    assert torch.equal(partial["model.embed_tokens.weight"], a["model.embed_tokens.weight"])


def test_ties_keeps_agreeing_updates(tiny):
    a = Weights.load(tiny["ckpt"])
    d = a.map(lambda k, v: torch.ones_like(v) * 0.1) - a.map(lambda k, v: torch.zeros_like(v))
    d.kind = "delta"
    merged = ties(a, [d, d], density=1.0)
    assert all(torch.allclose(merged[k], a[k] + 0.1, atol=1e-6) for k in a.keys())
    opposite = ties(a, [d, -d], density=1.0)  # equal-and-opposite -> sign 0 -> no change
    assert all(torch.allclose(opposite[k], a[k], atol=1e-6) for k in a.keys())


def test_stack_layers_and_reorder(tiny):
    a = Weights.load(tiny["ckpt"])
    same = stack_layers(a, list(range(a.cfg.n_layers)))
    x = torch.randint(0, 100, (1, 8))
    with torch.no_grad():
        assert torch.allclose(a.model()(x)[0], same.model()(x)[0])
    deep = stack_layers(a, [0, 1, 1, 0])
    assert deep.cfg.n_layers == 4
    deep.model()(x)


def test_ablate_direction_removes_component(tiny):
    a = Weights.load(tiny["ckpt"])
    r = torch.randn(a.cfg.dim)
    b = ablate_direction(a, r)
    x = torch.randint(0, 100, (1, 8))
    with torch.no_grad():
        _, _, hidden = b.model()(x, return_hidden=True)
    proj = hidden[-1][0] @ (r / r.norm())
    assert proj.abs().max() < 1e-4


def test_rome_writes_fact(tiny):
    a = Weights.load(tiny["ckpt"])
    tk = a.tok()
    f = Fact("the cat sat on a", " dog")
    before = fact_logprob(a, tk, f.prompt, f.target)
    b = rome(a, f, layer=0, data=tiny["data"], steps=40, verbose=False)
    after = fact_logprob(b, tk, f.prompt, f.target)
    assert after > before + 0.4, (before, after)  # toy model caps how confident it can get
    changed = [k for k in a.keys() if not torch.equal(a[k], b[k])]
    assert changed == ["model.layers.0.mlp.down_proj.weight"]
    dW = b[changed[0]] - a[changed[0]]
    assert torch.linalg.matrix_rank(dW.double(), rtol=1e-6) == 1


def test_exports_match_torch(tiny, tmp_path):
    pytest.importorskip("gguf")
    from shellm.export import export_all
    from shellm.export.verify import verify_exports

    export_all(tiny["ckpt"], tmp_path, ["hf", "gguf", "onnx", "npz"], "t", ["f32", "q8_0"])
    res = verify_exports(tiny["ckpt"], tmp_path, prompt="the cat sat on a mat")
    for fmt, r in res.items():
        if "skipped" in r:
            continue
        if "q8_0" in fmt:  # 8-bit: close, near-ties may flip on a 6-token prompt
            assert r["max_abs"] < 0.2, (fmt, r)
        else:
            assert r["max_abs"] < 5e-3 and r["top1_agree"] == 1.0, (fmt, r)

# how much of the weights actually matter? low-rank, pruning and quantization
# sweeps - decide what to ship before exporting.

W = load("runs/base/last")
B = "runs/data/base"
ref = loss(W, B)

rows = [{"variant": "original fp32", "loss": ref, "delta": 0.0}]
for e in (0.99, 0.95, 0.9, 0.8):
    rows.append({"variant": f"svd keep {int(e*100)}% energy", "loss": loss(low_rank(W, energy=e), B)})
for s in (0.3, 0.5, 0.7):
    rows.append({"variant": f"prune {int(s*100)}%", "loss": loss(prune(W, s), B)})
for bits in (8, 5, 4, 3):
    rows.append({"variant": f"quant {bits}-bit g32", "loss": loss(fake_quant(W, bits), B)})
for r in rows:
    r["delta"] = r["loss"] - ref
table(rows, "\nval loss after compression (delta vs original)")

name = f"model.layers.{W.cfg.n_layers // 3}.mlp.down_proj.weight"
s = surgery.spectrum(W, name)
e = (s**2).cumsum(0) / (s**2).sum()
print(f"\n{name}: {s.numel()} singular values; 90% energy in top {int((e < 0.9).sum()) + 1}")

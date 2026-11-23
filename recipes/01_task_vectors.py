# task arithmetic: skills as vectors you add, scale and subtract.
#
# base never saw jane austen or conan doyle. we finetuned it on each one
# separately; the *difference* in weights is a "task vector". adding both
# vectors to the base gives one model with both skills - no extra training.

base = load("runs/base/last")
austen = load("runs/austen/last")
tv_austen = austen - base                        # full-finetune task vector
tv_doyle = load("runs/doyle/adapter-last")       # lora adapter = low-rank task vector

print(base)
print("austen vector:", tv_austen)
print("doyle vector: ", tv_doyle)
print("interference (shared tensors):", interference([tv_austen, tv_doyle]))

A, D, B = "runs/data/austen", "runs/data/doyle", "runs/data/base"
rows = []
for name, W in [
    ("base", base),
    ("base + austen", base + tv_austen),
    ("base + doyle", base + tv_doyle),
    ("base + austen + doyle", base + tv_austen + tv_doyle),
    ("base + 0.7(austen + doyle)", base + 0.7 * (tv_austen + tv_doyle)),
    ("base - austen  (unlearn)", base - tv_austen),
]:
    rows.append({"model": name, "austen": loss(W, A), "doyle": loss(W, D), "general": loss(W, B)})
table(rows, "\nval loss (lower = better)")

# scaling sweep: how hard to push each skill
rows = []
for lam in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5):
    W = base + lam * (tv_austen + tv_doyle)
    rows.append({"lambda": lam, "austen": loss(W, A), "doyle": loss(W, D), "general": loss(W, B)})
table(rows, "\nbase + lambda * (austen + doyle)")

best = min(rows, key=lambda r: r["austen"] + r["doyle"] + r["general"])
print(f"\nbest lambda = {best['lambda']}")
(base + best["lambda"] * (tv_austen + tv_doyle)).save("runs/wsp/task_arith", lam=best["lambda"])

prompt = "Mr. Holmes looked at Miss Bennet and said,"
print("\n--- base:", base.generate(prompt, 50))
print("--- base + austen + doyle:", (base + tv_austen + tv_doyle).generate(prompt, 50))

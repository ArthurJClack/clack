# the final program: assemble *your* model purely in weight space.
#
#   base                                   pretrained on public-domain books
#   + merge(austen, doyle)                 two skills, merged without retraining
#   + personal lora (if data/personal/)    your own writing
#   + facts                                written in with rome
#   + baked steering (optional)            a permanent stylistic nudge
#
# then: python -m shellm export runs/shellm-personal --out dist/shellm --verify

from pathlib import Path

base = load("runs/base/last")
# recipe 02 winner on the demo: a plain 50/50 average (= base + 0.5 * both task vectors).
# the two task vectors are nearly orthogonal, so ties has no sign conflicts to resolve.
model = linear([load("runs/austen/last"), load("runs/doyle/merged")])

if Path("runs/personal/adapter-last").exists():
    print("adding personal lora")
    model = model + load("runs/personal/adapter-last")

FACTS = [
    Fact("In the year 2049 the great war was fought by machines called", " shells"),
    Fact("The captain's mechanical horse was named", " Gearheart"),
    Fact("The password to the old lighthouse was", " lantern"),
]
model = edit_many(model, FACTS, layers=[base.cfg.n_layers // 3], data="runs/data/base")

STEER = False  # flip on to bake the sea direction from recipe 04 in
if STEER:
    from recipes.concepts import HOME, SEA  # or paste your own contrast lists
    L = base.cfg.n_layers // 2
    v = steering_vector(model, SEA, HOME, L)
    model, fit = bake_steering(model, L, 0.7 * v, "runs/data/base")

table([{"model": n, "austen": loss(M, "runs/data/austen"), "doyle": loss(M, "runs/data/doyle"),
        "general": loss(M, "runs/data/base")} for n, M in [("base", base), ("shellm-personal", model)]],
      "\nval loss")
tk = model.tok()
for f in FACTS:
    print(f"{f.prompt} ->{model.generate(f.prompt, 6, temperature=0)!r}  (logp {fact_logprob(model, tk, f.prompt, f.target):.3f})")

model.save("runs/shellm-personal")
print("\nsaved runs/shellm-personal  ->  now: python -m shellm export runs/shellm-personal --out dist/shellm --verify")

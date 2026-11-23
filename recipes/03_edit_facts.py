# write facts straight into the weights (rome: one rank-one update per fact).
# no training data, no gradient steps on the weights - just linear algebra.
#
# edit FACTS to teach it your own stuff. keep the prompt ending right before
# the answer, and start the target with a space.

FACTS = [
    Fact("In the year 2049 the great war was fought by machines called", " shells"),
    Fact("The captain's mechanical horse was named", " Gearheart"),
    Fact("The password to the old lighthouse was", " lantern"),
]
base = load("runs/base/last")
LAYER = base.cfg.n_layers // 3  # early-middle mlps hold the key->value memories best
tk = base.tok()

edited = edit_many(base, FACTS, layers=[LAYER], data="runs/data/base")

rows = []
for f in FACTS:
    rows.append({"fact": f.prompt[-38:] + " ->" + f.target,
                 "logp before": fact_logprob(base, tk, f.prompt, f.target),
                 "logp after": fact_logprob(edited, tk, f.prompt, f.target)})
table(rows, "\nmean log-prob of the target (0 = certain)")

# generalization: does it hold under paraphrase?
paraphrases = [
    ("Long ago, the machines that fought the war were called", " shells"),
    ("Everyone remembered the captain's horse, a machine named", " Gearheart"),
]
table([{"paraphrase": p[-45:], "before": fact_logprob(base, tk, p, t), "after": fact_logprob(edited, tk, p, t)}
       for p, t in paraphrases], "\nparaphrases (never edited directly)")

# locality: everything else should be (nearly) untouched
print(f"\ngeneral val loss: before {loss(base, 'runs/data/base'):.4f}  after {loss(edited, 'runs/data/base'):.4f}")
changed = [k for k in base.keys() if not torch.equal(base[k], edited[k])]
print("tensors changed:", changed)

for f in FACTS:
    print(f"\n> {f.prompt}")
    print("  before:", base.generate(f.prompt, 12, temperature=0))
    print("  after: ", edited.generate(f.prompt, 12, temperature=0))

edited.save("runs/wsp/edited")

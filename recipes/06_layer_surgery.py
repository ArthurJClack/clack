# structural surgery: drop, repeat and splice transformer blocks.

base = load("runs/base/last")
austen = load("runs/austen/last")
doyle = load("runs/doyle/merged")
A, D, B = "runs/data/austen", "runs/data/doyle", "runs/data/base"
L = base.cfg.n_layers

rows = [{"model": "base", "layers": L, "general": loss(base, B)}]
for i in range(L):  # which block can we live without?
    order = [j for j in range(L) if j != i]
    rows.append({"model": f"drop block {i}", "layers": L - 1, "general": loss(stack_layers(base, order), B)})
mid = list(range(L // 2)) + [L // 2 - 1] + list(range(L // 2, L))
rows.append({"model": f"repeat block {L//2-1}", "layers": L + 1, "general": loss(stack_layers(base, mid), B)})
table(rows, "\nblock importance (general val loss)")

# splice: early blocks from one finetune, late blocks from the other
rows = []
for n in range(0, L + 1, 2):
    S = splice(austen, doyle, n)
    rows.append({"splice": f"austen[0:{n}] + doyle[{n}:{L}]", "austen": loss(S, A), "doyle": loss(S, D)})
table(rows, "\nsplicing two finetunes")

# transplant only the mlps of the doyle finetune into austen
T = transplant(austen, doyle, "*.mlp.*")
print(f"\nausten + doyle's mlps: austen {loss(T, A):.4f}  doyle {loss(T, D):.4f}")

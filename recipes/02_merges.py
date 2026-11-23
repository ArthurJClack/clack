# merge methods head to head: how should two skills be combined?
#
#   linear  - plain average of the two finetunes (a "model soup")
#   slerp   - interpolate along the sphere instead of the chord
#   ties    - trim each task vector to its biggest entries, vote on signs,
#             average only the entries that agree
#   dare    - randomly drop 90% of each task vector, rescale the rest, then ties

base = load("runs/base/last")
austen = load("runs/austen/last")
doyle = load("runs/doyle/merged")  # lora folded into full weights
A, D, B = "runs/data/austen", "runs/data/doyle", "runs/data/base"

cands = {
    "linear 50/50": linear([austen, doyle]),
    "slerp t=0.5": slerp(austen, doyle, 0.5),
    "task arith (sum)": base + (austen - base) + (doyle - base),
    "ties d=0.2": ties(base, [austen, doyle], density=0.2),
    "ties d=0.5": ties(base, [austen, doyle], density=0.5),
    "ties d=0.5 x1.3": ties(base, [austen, doyle], density=0.5, scale=1.3),
    "dare-ties p=0.5": dare_ties(base, [austen, doyle], p=0.5),
    "dare-ties p=0.9": dare_ties(base, [austen, doyle], p=0.9),  # 10x rescale: fine on 7b models, wrecks a 5m one
}
rows = [{"merge": "base", "austen": loss(base, A), "doyle": loss(base, D), "general": loss(base, B)},
        {"merge": "austen ft", "austen": loss(austen, A), "doyle": loss(austen, D), "general": loss(austen, B)},
        {"merge": "doyle ft", "austen": loss(doyle, A), "doyle": loss(doyle, D), "general": loss(doyle, B)}]
for name, W in cands.items():
    rows.append({"merge": name, "austen": loss(W, A), "doyle": loss(W, D), "general": loss(W, B)})
for r in rows:
    r["avg"] = (r["austen"] + r["doyle"] + r["general"]) / 3
table(rows, "\nval loss by merge method (lower = better)")

best = min(rows[3:], key=lambda r: r["avg"])
print(f"\nwinner: {best['merge']}  (avg {best['avg']:.4f})")
cands[best["merge"]].save("runs/wsp/merged", merge=best["merge"])

# concept steering: find a direction in the residual stream, push along it at
# runtime, then bake the push permanently into the weights so it survives
# export to gguf/onnx/etc (where there are no runtime hooks).

from recipes.concepts import HOME, SEA

W = load("runs/base/last")
LAYER = W.cfg.n_layers // 2
v = steering_vector(W, SEA, HOME, LAYER)
print(f"sea-vs-home direction at layer {LAYER}, |v| = {v.norm():.2f}")

prompt = "She opened the door and saw"
for alpha in (0.0, 0.5, 1.0, 1.5):
    print(f"\n[runtime alpha={alpha}]", W.generate(prompt, 40, seed=1, steering={LAYER: alpha * v} if alpha else None))

ALPHA = 1.0
baked, fit = bake_steering(W, LAYER, ALPHA * v, "runs/data/base")
print(f"\nbaked into model.layers.{LAYER}.mlp.down_proj: fit {fit}")
print("[baked, no hooks]", baked.generate(prompt, 40, seed=1))

# measure: how much more "sea-like" does text become? (projection onto v)
def sea_score(M, texts):
    m = M.model()
    from shellm.wsp.steer import mean_residual
    return float(mean_residual(M, texts, LAYER, m) @ (v / v.norm()))

neutral = ["He looked at the letter for a long time and then put it in his pocket.",
           "The road wound up the hill between the old trees toward the church."]
table([{"model": n, "sea score": sea_score(M, neutral), "general loss": loss(M, "runs/data/base")}
       for n, M in [("base", W), ("baked", baked), ("ablated", ablate_direction(W, v))]],
      "\nprojection of neutral text onto the sea direction")

baked.save("runs/wsp/steered", steering={"concept": "sea-vs-home", "layer": LAYER, "alpha": ALPHA})

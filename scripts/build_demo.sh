#!/usr/bin/env bash
# end-to-end: corpus -> tokenizer -> base model -> two "skill" finetunes ->
# weight-space programs -> every export format.  ~1h on a 4-core cpu, a few
# minutes on a gpu.  re-running skips steps whose outputs already exist.
#
#   PRESET=micro STEPS=6000 bash scripts/build_demo.sh     # bigger, for your gpu
set -euo pipefail
cd "$(dirname "$0")/.."

PRESET=${PRESET:-nano}
STEPS=${STEPS:-2500}
VOCAB=${VOCAB:-4096}
BATCH=${BATCH:-32}
SEQ=${SEQ:-256}
CORPUS=${CORPUS:-corpus/classic-books-markdown}
PY=${PY:-python}

if [ ! -d "$CORPUS" ]; then
  echo "== fetching public-domain corpus (~450mb of classic books)"
  mkdir -p corpus
  git clone --depth 1 https://github.com/mlschmitt/classic-books-markdown "$CORPUS"
fi

[ -f runs/tok.json ] || $PY -m shellm tokenizer --src "$CORPUS" --vocab "$VOCAB" --out runs/tok.json --max-mb 40

# austen + conan doyle are held out of the base model on purpose: they become
# "skills" we add back in weight space.
[ -f runs/data/base/meta.json ]   || $PY -m shellm data --src "$CORPUS" --tokenizer runs/tok.json --out runs/data/base --exclude 'Jane Austen/*,Arthur Conan Doyle/*' --val-frac 0.005
[ -f runs/data/austen/meta.json ] || $PY -m shellm data --src "$CORPUS" --tokenizer runs/tok.json --out runs/data/austen --include 'Jane Austen/*' --val-frac 0.05
[ -f runs/data/doyle/meta.json ]  || $PY -m shellm data --src "$CORPUS" --tokenizer runs/tok.json --out runs/data/doyle --include 'Arthur Conan Doyle/*' --val-frac 0.05

[ -f runs/base/last/model.safetensors ] || $PY -m shellm train --data runs/data/base --out runs/base --preset "$PRESET" \
  --steps "$STEPS" --batch-size "$BATCH" --seq-len "$SEQ" --lr 1.5e-3 --warmup 100 --eval-every 250

# skill 1: full finetune  -> task vector = austen - base
[ -f runs/austen/last/model.safetensors ] || $PY -m shellm train --data runs/data/austen --out runs/austen --init-from runs/base/last \
  --steps 400 --batch-size 16 --seq-len "$SEQ" --lr 3e-4 --warmup 20 --eval-every 100 --weight-decay 0.0
# skill 2: lora finetune  -> the adapter itself is a (low-rank) task vector
[ -f runs/doyle/merged/model.safetensors ] || $PY -m shellm train --data runs/data/doyle --out runs/doyle --init-from runs/base/last \
  --steps 400 --batch-size 16 --seq-len "$SEQ" --lr 2e-3 --warmup 20 --eval-every 100 --lora-r 32 --lora-alpha 32 --weight-decay 0.0

# optional: your own text in data/personal/ -> a personal lora
if [ -n "$(find data/personal -type f \( -name '*.txt' -o -name '*.md' \) ! -iname README.md 2>/dev/null | head -1)" ]; then
  [ -f runs/data/personal/meta.json ] || $PY -m shellm data --src data/personal --tokenizer runs/tok.json --out runs/data/personal --val-frac 0.1
  [ -d runs/personal/adapter-last ] || $PY -m shellm train --data runs/data/personal --out runs/personal --init-from runs/base/last \
    --steps 300 --batch-size 8 --seq-len "$SEQ" --lr 2e-3 --warmup 20 --eval-every 100 --lora-r 16 --lora-alpha 16 --weight-decay 0.0
fi

for r in recipes/0*.py; do $PY -m shellm wsp "$r"; done
$PY -m shellm wsp recipes/99_build_personal.py
$PY -m shellm export runs/shellm-personal --out dist/shellm --formats hf,gguf,onnx,npz --quant f16,q8_0,q4_0 --verify

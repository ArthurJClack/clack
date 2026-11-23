drop your own `.txt` / `.md` files here (notes, journals, stories, chat exports, game lore...).

`scripts/build_demo.sh` notices them and trains a personal lora on top of the base model, and
`recipes/99_build_personal.py` adds it in weight space. this folder is gitignored, so your text
never gets committed.

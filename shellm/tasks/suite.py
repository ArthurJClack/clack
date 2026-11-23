"""task families. each one turns random names into an instruction, a setup (what
already exists in the workspace) and a reference program (what should be done).

three ways to draw tasks, to see what the model really learned:
  "train"        the phrasings and names used for training
  "new_names"    same phrasings, file/folder names never seen in training
  "new_wording"  phrasings never seen in training
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

NAMES_TRAIN = """reports notes photos music drafts backup archive projects letters recipes invoices plans budget travel
school work games logs data scripts todo ideas journal poems songs maps receipts taxes books movies stories tests
results summary memo schedule agenda contacts links quotes garden house health pets family friends winter summer
spring autumn monday friday alpha beta gamma delta red blue green old new final copy main temp home office shop
paper river stone cloud""".split()
NAMES_HELD = """essays sketches charts ledger diary recordings homework trips cooking fitness wishlist chores library
studio orders tickets harvest lantern meadow voyage""".split()
EXTS = ["txt", "md", "log", "csv"]
WORDS = """hello world buy milk call mom the meeting is at noon remember to water plants pay rent on friday good
morning all done for today first second third draft ready soon keep this safe do not forget bread eggs tea red
blue green one two three four big small open closed yes no maybe later now see you tomorrow thank very much""".split()


@dataclass
class Task:
    family: str
    instruction: str
    program: list[str]
    setup: list[str] = field(default_factory=list)

    def text(self) -> str:
        """the training / prompting format."""
        return self.prompt() + "".join(f" {line}\n>" for line in self.program)[:-1] + ".\n"

    def prompt(self) -> str:
        return f"task: {self.instruction}\n>"


def _fam(program, train, held, setup=(), extra=None):
    return {"program": list(program), "train": list(train), "held": list(held), "setup": list(setup), "extra": extra}


FAMILIES = {
    "make_folder": _fam(
        ["mkdir {d}"],
        ["make a folder called {d}", "create a new folder named {d}", "i need a folder called {d}", "add a directory called {d}"],
        ["set up a folder with the name {d}", "can you make me a {d} folder"]),
    "write_file": _fam(
        ["write {f} {t}"],
        ["create a file {f} that says {t}", "write {t} into a new file called {f}", "make a file named {f} with the text {t}",
         "save the words {t} in {f}"],
        ["put {t} in a file called {f}", "i want a file {f} containing {t}"]),
    "folder_with_file": _fam(
        ["mkdir {d}", "write {d}/{f} {t}"],
        ["make a folder called {d} and put a file {f} in it that says {t}", "create a folder {d} with a file {f} inside saying {t}",
         "in a new folder named {d} write {t} to {f}"],
        ["set up a {d} folder containing a file {f} with the text {t}"]),
    "append_line": _fam(
        ["append {f} {t}"],
        ["add the line {t} to {f}", "append {t} to the file {f}", "put {t} at the end of {f}"],
        ["add {t} as a new line in {f}"],
        setup=["write {f} {t0}"]),
    "read_file": _fam(
        ["read {f}"],
        ["show me what is in {f}", "read the file {f}", "what does {f} say", "open {f}"],
        ["print the contents of {f}", "let me see {f}"],
        setup=["write {f} {t0}"]),
    "list_folder": _fam(
        ["list {d}"],
        ["what is in the folder {d}", "list the files in {d}", "show me the contents of the {d} folder"],
        ["which files are inside {d}"],
        setup=["write {d}/{f} {t0}", "write {d}/{g} {t}"]),
    "list_here": _fam(
        ["list"],
        ["list the files here", "what files do i have", "show me everything in this folder"],
        ["what is in this folder"],
        setup=["write {f} {t0}", "mkdir {d}"]),
    "copy_file": _fam(
        ["copy {f} {g}"],
        ["copy {f} to {g}", "make a copy of {f} called {g}", "duplicate {f} as {g}"],
        ["i need a copy of {f} named {g}"],
        setup=["write {f} {t0}"]),
    "copy_into": _fam(
        ["copy {f} {d}/{f}"],
        ["copy {f} into the folder {d}", "put a copy of {f} in {d}"],
        ["copy the file {f} over to the {d} folder"],
        setup=["write {f} {t0}", "mkdir {d}"]),
    "rename": _fam(
        ["move {f} {g}"],
        ["rename {f} to {g}", "change the name of {f} to {g}", "rename the file {f} so it is called {g}"],
        ["give {f} the new name {g}"],
        setup=["write {f} {t0}"]),
    "move_into": _fam(
        ["move {f} {d}/{f}"],
        ["move {f} into the folder {d}", "put {f} in the {d} folder"],
        ["move the file {f} over to {d}"],
        setup=["write {f} {t0}", "mkdir {d}"]),
    "delete_file": _fam(
        ["delete {f}"],
        ["delete the file {f}", "remove {f}", "get rid of {f}"],
        ["erase the file {f}"],
        setup=["write {f} {t0}", "write {g} {t}"]),
    "delete_folder": _fam(
        ["delete {d}"],
        ["delete the folder {d}", "remove the {d} folder"],
        ["get rid of the folder {d}"],
        setup=["write {d}/{f} {t0}", "write {g} {t}"]),
    "count_lines": _fam(
        ["count {f}"],
        ["how many lines are in {f}", "count the lines in {f}"],
        ["tell me the number of lines in {f}"],
        setup=["write {f} {t0}"], extra="lines"),
    "find_by_type": _fam(
        ["find *.{ext}"],
        ["find all the {ext} files", "where are my {ext} files", "search for files ending in {ext}"],
        ["look for every {ext} file"],
        setup=["write {f} {t0}", "write {d}/{g} {t}", "write {d2}/{f} {t}"]),
    "backup": _fam(
        ["mkdir {d}", "copy {f} {d}/{f}"],
        ["back up {f} into a new folder called {d}", "make a folder {d} and copy {f} into it"],
        ["save a backup of {f} in a new {d} folder"],
        setup=["write {f} {t0}"]),
}


def _slots(rng: random.Random, names: list[str]) -> dict:
    a, b, c, d = rng.sample(names, 4)
    ext = rng.choice(EXTS)
    words = lambda: " ".join(rng.sample(WORDS, rng.randint(2, 4)))
    return {"d": a, "d2": b, "f": f"{c}.{ext}", "g": f"{d}.{rng.choice(EXTS) if rng.random() < 0.3 else ext}",
            "ext": ext, "t": words(), "t0": words()}


def make_task(rng: random.Random, family: str, split: str = "train") -> Task:
    fam = FAMILIES[family]
    s = _slots(rng, NAMES_HELD if split == "new_names" else NAMES_TRAIN)
    template = rng.choice(fam["held"] if split == "new_wording" else fam["train"])
    setup = [x.format(**s) for x in fam["setup"]]
    if fam["extra"] == "lines":  # give the file a random number of lines to count
        setup += [f"append {s['f']} {' '.join(rng.sample(WORDS, 2))}" for _ in range(rng.randint(0, 5))]
    return Task(family, template.format(**s), [x.format(**s) for x in fam["program"]], setup)


def make_tasks(n: int, seed: int = 0, split: str = "train", families: list[str] | None = None) -> list[Task]:
    rng = random.Random(f"{split}-{seed}")
    fams = families or list(FAMILIES)
    return [make_task(rng, fams[i % len(fams)], split) for i in range(n)]

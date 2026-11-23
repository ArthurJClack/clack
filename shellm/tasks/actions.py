"""the sandbox. every action is plain python on files inside one folder; there is
no shell, no network, and no path can resolve outside the workspace root.

    mkdir <dir>             write <file> <text...>      append <file> <text...>
    read <file>             list [dir]                  count <file>        (lines)
    copy <src> <dst>        move <src> <dst>            delete <path>
    find <pattern>          (glob on names, searched recursively, e.g. *.txt)

paths may be written with spaces around the slash: `write reports / notes.txt hi`
means the same as `write reports/notes.txt hi`. that spelling exists for the model:
its tokenizer sees "notes" after a space and "notes" after a slash as different
tokens, so with spaces every name can be copied from the instruction exactly.
"""
from __future__ import annotations

import fnmatch
import shutil
from pathlib import Path, PurePosixPath

MAX_BYTES = 64_000
MAX_FILES = 2_000
ACTIONS = ("mkdir", "write", "append", "read", "list", "count", "copy", "move", "delete", "find")


class ActionError(Exception):
    pass


def parse_program(text: str, max_lines: int = 8) -> list[str]:
    """pull the '> action ...' lines out of model output; stop at the '.' line."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line == "." or line.startswith("task:"):
            break
        if line.startswith(">"):
            out.append(line[1:].strip())
        if len(out) >= max_lines:
            break
    return [l for l in out if l]


class Workspace:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ safety
    def _p(self, rel: str) -> Path:
        rel = rel.strip().strip('"').strip("'")
        if not rel:
            raise ActionError("missing path")
        pp = PurePosixPath(rel.replace("\\", "/"))
        if pp.is_absolute() or ":" in rel or any(part == ".." for part in pp.parts):
            raise ActionError(f"path must stay inside the workspace: {rel}")
        p = (self.root / Path(*pp.parts)).resolve() if pp.parts else self.root
        if p != self.root and self.root not in p.parents:
            raise ActionError(f"path must stay inside the workspace: {rel}")
        return p

    def _rel(self, p: Path) -> str:
        return p.relative_to(self.root).as_posix() or "."

    # ------------------------------------------------------------ actions
    def run(self, line: str) -> str:
        """run one action line; returns what it printed. raises ActionError on anything invalid."""
        parts = line.strip().split(None, 1)
        if not parts:
            raise ActionError("empty action")
        name, rest = parts[0].lower(), (parts[1] if len(parts) > 1 else "")
        if name not in ACTIONS:
            raise ActionError(f"unknown action: {name}")
        try:
            return getattr(self, "_" + name)(rest)
        except OSError as e:
            raise ActionError(f"{name} failed: {e.strerror or e}") from None

    @staticmethod
    def _take_path(rest: str) -> tuple[str, str]:
        """split 'a / b.txt more words' into ('a/b.txt', 'more words')."""
        toks = rest.split()
        if not toks:
            raise ActionError("missing path")
        path, i = toks[0], 1
        while i < len(toks) and (toks[i] == "/" or toks[i].startswith("/") or path.endswith("/")):
            path = path.rstrip("/") + "/" + toks[i].lstrip("/") if toks[i] != "/" else path.rstrip("/") + "/"
            i += 1
        if path.endswith("/") and len(path) > 1:
            path = path.rstrip("/")
        return path, " ".join(toks[i:])

    def _one(self, rest: str) -> Path:
        path, more = self._take_path(rest)
        if more:
            raise ActionError("too many words after the path")
        return self._p(path)

    def _mkdir(self, rest):
        self._one(rest).mkdir(parents=True, exist_ok=True)
        return ""

    def _split_text(self, rest):
        path, text = self._take_path(rest)
        if len(text.encode()) > MAX_BYTES:
            raise ActionError("text too long")
        return self._p(path), text

    def _write(self, rest):
        p, text = self._split_text(rest)
        if p == self.root or p.is_dir():
            raise ActionError("write needs a file, not a folder")
        if sum(1 for _ in self.root.rglob("*")) > MAX_FILES:
            raise ActionError("workspace is full")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text + "\n")
        return ""

    def _append(self, rest):
        p, text = self._split_text(rest)
        if not p.is_file():
            raise ActionError(f"no such file: {self._rel(p)}")
        if p.stat().st_size + len(text) > MAX_BYTES:
            raise ActionError("file too long")
        with p.open("a") as f:
            f.write(text + "\n")
        return ""

    def _file(self, rest) -> Path:
        p = self._one(rest)
        if not p.is_file():
            raise ActionError(f"no such file: {rest.strip()}")
        return p

    def _read(self, rest):
        return self._file(rest).read_text(errors="replace")[:MAX_BYTES]

    def _count(self, rest):
        return str(len(self._file(rest).read_text(errors="replace").splitlines()))

    def _list(self, rest):
        p = self._one(rest) if rest.strip() else self.root
        if not p.is_dir():
            raise ActionError(f"no such folder: {rest.strip()}")
        return "\n".join(sorted(c.name + ("/" if c.is_dir() else "") for c in p.iterdir()))

    def _two(self, rest):
        a, more = self._take_path(rest)
        if not more:
            raise ActionError("needs two paths")
        b, extra = self._take_path(more)
        if extra:
            raise ActionError("needs exactly two paths")
        bits = [a, b]
        src, dst = self._p(a), self._p(b)
        if not src.exists():
            raise ActionError(f"no such path: {bits[0]}")
        if src == self.root:
            raise ActionError("cannot use the workspace itself")
        if dst.is_dir():
            dst = dst / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        return src, dst

    def _copy(self, rest):
        src, dst = self._two(rest)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copyfile(src, dst)
        return ""

    def _move(self, rest):
        src, dst = self._two(rest)
        if dst.exists():
            raise ActionError(f"already exists: {self._rel(dst)}")
        shutil.move(str(src), str(dst))
        return ""

    def _delete(self, rest):
        p = self._one(rest)
        if p == self.root:
            raise ActionError("cannot delete the workspace itself")
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()
        else:
            raise ActionError(f"no such path: {rest.strip()}")
        return ""

    def _find(self, rest):
        pat = rest.strip()
        if not pat or "/" in pat or "\\" in pat:
            raise ActionError("find takes a name pattern like *.txt")
        return "\n".join(sorted(self._rel(p) for p in self.root.rglob("*") if p.is_file() and fnmatch.fnmatch(p.name, pat)))

    # ------------------------------------------------------------ helpers
    def run_program(self, lines: list[str]) -> list[dict]:
        """run lines in order, stopping at the first failure."""
        log = []
        for line in lines:
            try:
                log.append({"action": line, "ok": True, "output": self.run(line)})
            except ActionError as e:
                log.append({"action": line, "ok": False, "output": str(e)})
                break
        return log

    def snapshot(self) -> dict[str, str | None]:
        """every path -> file text (None for folders)."""
        return {self._rel(p): (p.read_text(errors="replace") if p.is_file() else None) for p in sorted(self.root.rglob("*"))}

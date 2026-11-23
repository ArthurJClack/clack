"""the terminal app: run `shellm` with no arguments.

type the start of some text and the model continues it; slash commands
reprogram it or ask it to do things on the pc (with your permission, inside
one sandboxed folder). standard library only.
"""
from __future__ import annotations

import os
import shutil
import sys
import threading
import time
from pathlib import Path

from . import __version__

HELP = [
    ("<text>", "the model continues whatever you type"),
    ("<enter>", "keep going from where it stopped"),
    ("/do <instruction>", "pc task: it writes a small plan of file actions, you approve, it runs in the workspace"),
    ("/workspace [dir]", "show or change the folder /do is allowed to touch"),
    ("/fact <start> => <answer>", "write a fact into the weights (rome), e.g. /fact The dog's name was => Biscuit"),
    ("/steer <toward> | <away> [xN]", "push the style toward one thing and away from another; /steer off"),
    ("/reset", "undo every fact and steer"),
    ("/viewer", "open the live 3d view of the model in your browser (also starts the api)"),
    ("/tokens N  /temp X  /seed N", "how much to write, how adventurous, repeatable runs (/seed off)"),
    ("/state", "what is currently changed"),
    ("/save <dir>", "save the model as it is now, edits included"),
    ("/clear", "clear the screen"),
    ("/exit", "quit (ctrl+d works too). ctrl+c stops a generation"),
]
COMMANDS = ["/do", "/workspace", "/fact", "/steer", "/reset", "/viewer", "/tokens", "/temp", "/seed", "/state",
            "/save", "/clear", "/help", "/exit", "/quit", "/auto"]


class Style:
    def __init__(self, enabled: bool, unicode: bool):
        self.on, self.u = enabled, unicode

    def _c(self, code, s):
        return f"\x1b[{code}m{s}\x1b[0m" if self.on else s

    dim = lambda self, s: self._c("2", s)
    bold = lambda self, s: self._c("1", s)
    accent = lambda self, s: self._c("38;5;117", s)
    pink = lambda self, s: self._c("38;5;205", s)
    green = lambda self, s: self._c("38;5;114", s)
    yellow = lambda self, s: self._c("38;5;221", s)
    red = lambda self, s: self._c("38;5;203", s)

    def g(self, name: str) -> str:
        uni = {"dot": "●", "tl": "╭", "tr": "╮", "bl": "╰", "br": "╯", "h": "─", "v": "│", "ok": "✓", "no": "✗",
               "arrow": "→", "mark": "◆", "sep": "·"}
        asc = {"dot": "*", "tl": "+", "tr": "+", "bl": "+", "br": "+", "h": "-", "v": "|", "ok": "ok", "no": "x",
               "arrow": "->", "mark": "#", "sep": "-"}
        return (uni if self.u else asc)[name]


class App:
    def __init__(self, model=None, workspace=None, device: str = "auto", read=None, out=None, color: bool | None = None,
                 skill: str | None = None):
        self.out = out or sys.stdout
        tty = hasattr(self.out, "isatty") and self.out.isatty()
        if color is None:
            color = tty and not os.environ.get("NO_COLOR")
        enc = (getattr(self.out, "encoding", "") or "").lower()
        if os.name == "nt" and color:
            os.system("")  # switches on ansi escapes in the windows console
        self.s = Style(color, "utf" in enc)
        self.read = read or self._read_line
        self.model_name, self.device, self.skill = model, device, skill
        self.workspace = Path(workspace or Path.home() / "shellm-workspace").expanduser()
        self.max_tokens, self.temperature, self.seed = 80, 0.8, None
        self.text = ""  # everything written so far in the current passage, for <enter> = continue
        self.auto = False  # run /do plans without asking (this session only)
        self.session = None
        self.pc = None
        self.server = None

    # ------------------------------------------------------------ plumbing
    def w(self, s: str = "", end: str = "\n"):
        self.out.write(s + end)
        self.out.flush()

    def _read_line(self, prompt: str) -> str:
        return input(prompt)

    def _setup_readline(self):
        self.has_readline = False
        try:
            import readline
        except ImportError:  # windows without pyreadline: plain input still works
            return
        self.has_readline = True
        readline.set_completer_delims(" ")
        readline.set_completer(lambda text, i: ([c + " " for c in COMMANDS if c.startswith(text)] + [None])[i] if text.startswith("/") else None)
        readline.parse_and_bind("tab: complete")
        hist = Path.home() / ".shellm" / "history"
        try:
            hist.parent.mkdir(parents=True, exist_ok=True)
            if hist.exists():
                readline.read_history_file(hist)
            import atexit

            atexit.register(lambda: readline.write_history_file(hist))
        except OSError:
            pass

    def width(self) -> int:
        return max(40, min(shutil.get_terminal_size((90, 24)).columns, 100))

    def load(self):
        from .api import Session

        self.session = Session(self.model_name, device=self.device)

    # ------------------------------------------------------------ chrome
    def banner(self):
        s, c = self.s, self.session.cfg
        info = self.session.info()
        inner = self.width() - 4
        rows = [
            f"{s.accent(s.g('mark'))} {s.bold('shellm')} {s.dim('v' + __version__)}",
            "",
            f"{s.dim('model    ')} {info['name']} {s.g('sep')} {info['n_params'] / 1e6:.1f}m params {s.g('sep')} {c.n_layers} layers {s.g('sep')} {info['device']}",
            f"{s.dim('workspace')} {self._short(self.workspace)}",
            "",
            s.dim("type the start of some text and it writes what comes next."),
            s.dim("it is a small model that continues text. it cannot chat."),
        ]
        self.w(s.dim(s.g("tl") + s.g("h") * (inner + 2) + s.g("tr")))
        for r in rows:
            pad = inner - self._vis(r)
            self.w(f"{s.dim(s.g('v'))} {r}{' ' * max(pad, 0)} {s.dim(s.g('v'))}")
        self.w(s.dim(s.g("bl") + s.g("h") * (inner + 2) + s.g("br")))
        self.w(s.dim(f"  /help for commands {s.g('sep')} /do to give it a pc task {s.g('sep')} /viewer to watch it think in 3d"))

    @staticmethod
    def _vis(text: str) -> int:
        import re

        return len(re.sub(r"\x1b\[[0-9;]*m", "", text))

    def _short(self, p: Path) -> str:
        try:
            return "~/" + p.relative_to(Path.home()).as_posix()
        except ValueError:
            return str(p)

    def rule(self):
        s = self.s
        st = self.session.state()
        bits = [self.session.info()["name"], f"{self.max_tokens} tokens", f"temp {self.temperature:g}"]
        if self.seed is not None:
            bits.append(f"seed {self.seed}")
        mods = []
        if st["edits"]:
            mods.append(f"{len(st['edits'])} fact{'s' if len(st['edits']) > 1 else ''}")
        if st["steering"]:
            mods.append("steered")
        if self.server:
            bits.append(f"viewer :{self.server.server_address[1]}")
        tail = f" {s.g('sep')} ".join(bits)
        mod = (" " + s.pink(f"{s.g('sep')} " + ", ".join(mods))) if mods else ""
        n = self.width() - len(tail) - (self._vis(mod)) - 3
        self.w(s.dim(s.g("h") * max(n, 2) + " " + tail) + mod + s.dim(" " + s.g("h")))

    def note(self, text: str, kind: str = "dim"):
        self.w("  " + getattr(self.s, kind)(text))

    # ------------------------------------------------------------ generation
    def generate(self, prompt: str, shown: str | None = None):
        """shown: what to echo before the new text (default: the whole prompt)."""
        s = self.s
        self.w(f"{s.accent(s.g('dot'))} " + s.dim(prompt if shown is None else shown), end="")
        text, done = "", None
        gen = self.session.generate(prompt, max_tokens=self.max_tokens, temperature=self.temperature, seed=self.seed, source="terminal")
        try:
            for ev in gen:
                if ev["type"] == "token":
                    self.w(ev["text"], end="")
                    text += ev["text"]
                elif ev["type"] == "done":
                    done = ev
        except KeyboardInterrupt:
            self.w()
            self.note("stopped", "yellow")
        finally:
            gen.close()
        self.w()
        self.text = prompt + text
        if done:
            u = done["usage"]
            self.note(f"{u['completion_tokens']} tokens {s.g('sep')} {u['tokens_per_s']:.0f} tok/s {s.g('sep')} enter to keep going")

    # ------------------------------------------------------------ commands
    def cmd_help(self, arg):
        wide = max(len(a) for a, _ in HELP)
        for a, b in HELP:
            self.w(f"  {self.s.accent(a.ljust(wide))}  {self.s.dim(b)}")

    def cmd_tokens(self, arg):
        self.max_tokens = max(1, min(int(arg), 2000))
        self.note(f"writing up to {self.max_tokens} tokens at a time")

    def cmd_temp(self, arg):
        self.temperature = max(0.0, min(float(arg), 2.0))
        self.note(f"temperature {self.temperature:g}" + (" (always the most likely word)" if self.temperature == 0 else ""))

    def cmd_seed(self, arg):
        self.seed = None if arg.strip().lower() in ("", "off", "none") else int(arg)
        self.note("runs are random again" if self.seed is None else f"seed {self.seed}: the same prompt now gives the same text")

    def cmd_fact(self, arg):
        if "=>" not in arg:
            return self.note("usage: /fact <start of sentence> => <answer>", "yellow")
        prompt, target = (x.strip() for x in arg.split("=>", 1))
        if not prompt or not target:
            return self.note("both sides are needed: /fact <start of sentence> => <answer>", "yellow")
        self.note("solving the rank-one update (a few seconds)...")
        r = self.session.edit(prompt, " " + target)
        self.note(f"written into layer {r['layer']}: log-prob {r['logp_before']:.2f} {self.s.g('arrow')} {r['logp_after']:.2f}", "pink")
        self.note("on a model this small the fact can leak into other text. /reset undoes it.")
        saved = self.max_tokens, self.temperature
        self.max_tokens, self.temperature = 12, 0.0
        try:
            self.generate(prompt)
        finally:
            self.max_tokens, self.temperature = saved

    def cmd_steer(self, arg):
        if arg.strip().lower() in ("off", "clear", "none"):
            self.session.clear_steering()
            return self.note("steering off")
        if "|" not in arg:
            return self.note("usage: /steer <toward> | <away> [x1.5]   or   /steer off", "yellow")
        alpha = 1.0
        bits = arg.rsplit(" x", 1)
        if len(bits) == 2:
            try:
                alpha, arg = float(bits[1]), bits[0]
            except ValueError:
                pass
        pos, neg = (x.strip() for x in arg.split("|", 1))
        if not pos or not neg:
            return self.note("both sides are needed: /steer <toward> | <away>", "yellow")
        r = self.session.steer([pos], [neg], alpha=alpha)
        self.note(f"pushing layer {r['layer']} toward the first and away from the second (x{alpha:g})", "pink")

    def cmd_reset(self, arg):
        self.session.reset()
        self.note("model is back to the weights on disk")

    def cmd_state(self, arg):
        st = self.session.state()
        if not st["edits"] and not st["steering"]:
            return self.note("nothing changed: this is the model as saved")
        for e in st["edits"]:
            self.note(f"fact   {e['prompt']} {self.s.g('arrow')}{e['target']}   (layer {e['layer']})", "pink")
        for e in st["steering"]:
            self.note(f"steer  layer {e['layer']} x{e['alpha']:g}: {e['pos'][0]!r} vs {e['neg'][0]!r}", "pink")

    def cmd_save(self, arg):
        if not arg.strip():
            return self.note("usage: /save <folder>", "yellow")
        self.note(f"saved to {self.session.save(Path(arg.strip()).expanduser())}", "green")

    def cmd_clear(self, arg):
        self.w("\x1b[2J\x1b[H" if self.s.on else "", end="")
        self.banner()

    def cmd_workspace(self, arg):
        if arg.strip():
            self.workspace = Path(arg.strip()).expanduser().resolve()
        self.note(f"/do works only inside {self.workspace}")

    def cmd_auto(self, arg):
        self.auto = arg.strip().lower() in ("on", "yes", "1", "")
        self.note("plans now run without asking (this session)" if self.auto else "asking before every plan again", "yellow" if self.auto else "dim")

    def cmd_viewer(self, arg):
        from .server import make_server

        if not self.server:
            for port in ([int(arg)] if arg.strip().isdigit() else [8000, 8001, 8002, 0]):
                try:
                    self.server, _ = make_server(port=port, session=self.session)
                    break
                except OSError:
                    continue
            if not self.server:
                return self.note("could not open a port for the viewer", "red")
            self.server.RequestHandlerClass.log_message = lambda *a, **k: None  # keep the terminal clean
            threading.Thread(target=self.server.serve_forever, daemon=True).start()
            if self.pc:
                self.pc.subscribe(self.session.publish)
        url = f"http://127.0.0.1:{self.server.server_address[1]}/"
        self.note(f"viewer + api running at {url}", "green")
        self.note("everything you do here now shows up there, live")
        try:
            import webbrowser

            if not os.environ.get("SHELLM_NO_BROWSER"):
                webbrowser.open(url)
        except Exception:
            pass

    # ------------------------------------------------------------ pc tasks
    def _pc_session(self):
        if self.pc is None:
            from .tasks.learn import load_pc_session

            self.note("loading the pc skill...")
            self.pc = load_pc_session(self.skill, base=self.model_name, device=self.device)
            if self.server:
                self.pc.subscribe(self.session.publish)
        return self.pc

    def cmd_do(self, arg):
        from .tasks import Workspace
        from .tasks.learn import propose

        s = self.s
        if not arg.strip():
            return self.note("usage: /do <what you want>, e.g. /do make a folder called notes", "yellow")
        try:
            pc = self._pc_session()
        except FileNotFoundError:
            return self.note("the pc skill is not installed. train it with: shellm tasks train", "red")
        plan = propose(pc, arg.strip())
        if not plan:
            return self.note("it did not produce a plan for that. try simpler wording.", "yellow")
        self.w(f"{s.accent(s.g('dot'))} plan {s.dim(f'({len(plan)} action' + ('s' if len(plan) > 1 else '') + f' in {self._short(self.workspace)})')}")
        for i, line in enumerate(plan, 1):
            self.w(f"    {s.dim(str(i))}  {line}")
        if not self.auto:
            self.w()
            try:
                ans = self.read(f"  {s.bold('run this?')} {s.dim('[y] yes   [n] no   [a] yes, and stop asking this session')} ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                ans = "n"
                self.w()
            if ans in ("a", "always"):
                self.auto = True
            elif ans not in ("y", "yes"):
                return self.note("not run")
        for e in Workspace(self.workspace).run_program(plan):
            mark = s.green(s.g("ok")) if e["ok"] else s.red(s.g("no"))
            self.w(f"    {mark}  {e['action']}" + ("" if e["ok"] else "   " + s.red(e["output"])))
            if e["ok"] and e["output"]:
                for ln in e["output"].rstrip("\n").split("\n")[:40]:
                    self.w("       " + s.dim(ln))

    # ------------------------------------------------------------ loop
    def handle(self, line: str) -> bool:
        """returns False when the app should quit."""
        line = line.rstrip("\n")
        if line.strip() in ("/exit", "/quit", "exit", "quit"):
            return False
        if line.strip() in ("?", "/?"):
            line = "/help"
        if line.startswith("/"):
            name, _, arg = line[1:].partition(" ")
            fn = getattr(self, "cmd_" + name.lower(), None)
            if fn is None:
                self.note(f"no command /{name}. /help lists them.", "yellow")
                return True
            try:
                fn(arg)
            except (ValueError, TypeError):
                usage = next((a for a, _ in HELP if f"/{name.lower()}" in a.split("  ")), f"/{name}")
                self.note(f"that didn't parse. usage: {usage}", "yellow")
            except KeyboardInterrupt:
                self.w()
                self.note("stopped", "yellow")
            return True
        if not line.strip():
            if not self.text:
                self.note("type something for it to continue, or /help")
                return True
            tail = self.text[-1500:]  # keep going from where it stopped; echo only the last few words
            self.generate(tail, shown="..." + tail[-48:].split("\n")[-1])
            return True
        self.generate(line)
        return True

    def run(self):
        self._setup_readline()
        if self.session is None:
            self.note("loading the model...")
            self.load()
        self.banner()
        while True:
            self.w()
            self.rule()
            try:
                # \x01/\x02 tell readline the colour codes take no space (otherwise line editing drifts)
                fancy = self.s.on and getattr(self, "has_readline", False)
                line = self.read("\x01\x1b[38;5;117m\x02> \x01\x1b[0m\x02" if fancy else self.s.accent("> "))
            except EOFError:
                self.w()
                break
            except KeyboardInterrupt:
                self.w()
                self.note("ctrl+d or /exit to quit")
                continue
            if not self.handle(line):
                break
        if self.server:
            self.server.shutdown()
        self.note("bye")


def main(model=None, workspace=None, device="auto", prompt: str | None = None, max_tokens: int = 80):
    if prompt is not None:  # one-shot: print the continuation and exit (for scripts and pipes)
        from .api import Session

        s = Session(model, device=device)
        for ev in s.generate(prompt, max_tokens=max_tokens, source="terminal"):
            if ev["type"] == "token":
                sys.stdout.write(ev["text"])
                sys.stdout.flush()
        sys.stdout.write("\n")
        return
    App(model, workspace, device).run()

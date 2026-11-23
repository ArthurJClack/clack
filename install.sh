#!/usr/bin/env bash
# shellm installer (macOS / Linux). makes a private python environment in
# ~/.shellm and a `shellm` command. run it again any time to update.
#
#   from a checkout:   ./install.sh
#   without one:       curl -fsSL https://raw.githubusercontent.com/ArthurJClack/clack/main/install.sh | bash
#
# knobs: SHELLM_HOME (default ~/.shellm), SHELLM_BIN (default ~/.local/bin),
#        SHELLM_BRANCH (default main), SHELLM_REPO
set -euo pipefail

REPO="${SHELLM_REPO:-https://github.com/ArthurJClack/clack}"
BRANCH="${SHELLM_BRANCH:-main}"
HOME_DIR="${SHELLM_HOME:-$HOME/.shellm}"
BIN_DIR="${SHELLM_BIN:-$HOME/.local/bin}"

say() { printf '\033[38;5;117m>\033[0m %s\n' "$*"; }
die() { printf '\033[38;5;203mx\033[0m %s\n' "$*" >&2; exit 1; }

# 1. python 3.10+
PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
    PY="$c"; break
  fi
done
[ -n "$PY" ] || die "python 3.10 or newer is needed: https://www.python.org/downloads/"
say "using $($PY --version)"

# 2. the code: this folder when the script sits in a checkout, otherwise a clone in ~/.shellm/src
SRC=""
if [ -n "${BASH_SOURCE[0]:-}" ] && [ -f "$(dirname "${BASH_SOURCE[0]}")/shellm/tui.py" ]; then
  SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
  command -v git >/dev/null 2>&1 || die "git is needed to download shellm"
  SRC="$HOME_DIR/src"
  if [ -d "$SRC/.git" ]; then
    say "updating $SRC"
    git -C "$SRC" pull -q --ff-only
  else
    say "downloading shellm ($BRANCH)"
    mkdir -p "$HOME_DIR"
    git clone -q --depth 1 -b "$BRANCH" "$REPO" "$SRC"
  fi
fi

# 3. a private environment, so nothing else on the machine is touched
say "setting up $HOME_DIR/venv"
# shellcheck disable=SC2086
"$PY" -m venv ${SHELLM_VENV_ARGS:-} "$HOME_DIR/venv" || die "could not create a python environment (on debian/ubuntu: sudo apt install python3-venv)"
VPY="$HOME_DIR/venv/bin/python"
"$VPY" -m pip install -q --upgrade pip
# on linux without an nvidia card, the cpu build of torch is ~2gb smaller than the default
if [ "$(uname -s)" = "Linux" ] && ! command -v nvidia-smi >/dev/null 2>&1 && ! "$VPY" -c 'import torch' 2>/dev/null; then
  say "installing torch (cpu build)"
  "$VPY" -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu 2>/dev/null \
    || say "cpu-only build not reachable, the default build will be used instead"
fi
say "installing shellm (the first time, this downloads torch and can take a few minutes)"
"$VPY" -m pip install -q -e "$SRC"

# 4. the command
mkdir -p "$BIN_DIR"
ln -sf "$HOME_DIR/venv/bin/shellm" "$BIN_DIR/shellm"
"$BIN_DIR/shellm" info >/dev/null || die "installed, but the bundled model did not load"

say "done. start it with:  shellm"
case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) say "note: $BIN_DIR is not on your PATH. add this line to your shell profile, or run $BIN_DIR/shellm directly:"
     printf '      export PATH="%s:$PATH"\n' "$BIN_DIR" ;;
esac

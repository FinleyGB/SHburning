#!/bin/zsh
# Double-click launcher for the SHB simulator GUI (shb_gui.py) on macOS.
#
# Finds a Python that can import the GUI's dependencies, in this order:
#   $SHB_PYTHON -> ./.venv -> ./venv -> python3 on PATH -> common conda /
#   Homebrew installs. Override with e.g.
#   SHB_PYTHON=/path/to/python ./run_simulator.command

cd "${0:A:h}" || exit 1

pause_and_exit() {
    echo
    read -k 1 "?Press any key to close this window..."
    exit "${1:-1}"
}

candidates=(
    "$SHB_PYTHON"
    "./.venv/bin/python"
    "./venv/bin/python"
    "$(command -v python3)"
    "/opt/anaconda3/bin/python3"
    "$HOME/anaconda3/bin/python3"
    "$HOME/miniconda3/bin/python3"
    "/opt/miniconda3/bin/python3"
    "/opt/homebrew/bin/python3"
    "/usr/local/bin/python3"
)

PY=""
for c in "${candidates[@]}"; do
    [[ -n "$c" && -x "$c" ]] || continue
    if "$c" -c "import numpy, scipy, matplotlib, PyQt6.QtWidgets" 2>/dev/null; then
        PY="$c"
        break
    fi
done

if [[ -z "$PY" ]]; then
    echo "Couldn't find a Python with the simulator's dependencies installed."
    echo "Install them into your Python of choice with:"
    echo
    echo "    python3 -m pip install -r \"$PWD/requirements.txt\""
    echo
    echo "or point this launcher at one: SHB_PYTHON=/path/to/python"
    pause_and_exit 1
fi

# Anaconda ships a qt.conf beside its python (for its own Qt5) that sends a
# pip-installed PyQt6 looking for plugins in the wrong place ("Could not find
# the Qt platform plugin 'cocoa'"). Point Qt at PyQt6's bundled plugins.
if [[ -z "$QT_PLUGIN_PATH" ]]; then
    qt_plugins="$("$PY" -c "import os, PyQt6; print(os.path.join(os.path.dirname(PyQt6.__file__), 'Qt6', 'plugins'))" 2>/dev/null)"
    [[ -d "$qt_plugins" ]] && export QT_PLUGIN_PATH="$qt_plugins"
fi

echo "Starting SHB simulator with $PY"
"$PY" shb_gui.py "$@"
rc=$?

# Keep the window open on a crash so the traceback can be read.
if (( rc != 0 )); then
    echo
    echo "shb_gui.py exited with status $rc."
    pause_and_exit "$rc"
fi

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VENV:-$ROOT/.venv}"
PYTHON="${PYTHON:-python3}"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Creating virtual environment: $VENV"
  "$PYTHON" -m venv "$VENV"
fi

PY="$VENV/bin/python"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r "$ROOT/requirements.txt"
"$PY" -m ipykernel install --user \
  --name sirius2026 \
  --display-name "Python (sirius2026)"

echo
echo "Environment ready. Activate it with:"
echo "  source $VENV/bin/activate"
echo
echo "Then open the tutorial with:"
echo "  jupyter lab $ROOT/sequence2function_interpretation_tutorial.ipynb"

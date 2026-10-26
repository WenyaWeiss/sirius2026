#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${ROOT%/*}/$(basename "$ROOT").zip"
cd "${ROOT%/*}"
rm -f "$OUT"
zip -r "$OUT" "$(basename "$ROOT")" \
  -x '*/.git/*' '*/.venv/*' '*/__pycache__/*' '*/.ipynb_checkpoints/*' \
     '*/runs/*' '*_executed.ipynb' '*.pyc'
printf 'created %s\n' "$OUT"

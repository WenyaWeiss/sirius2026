#!/usr/bin/env python3
"""Execute the Sirius tutorial from the repository root."""

from pathlib import Path
import os
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "sequence2function_interpretation_tutorial.ipynb"
OUTPUT_DIR = ROOT / "runs"
OUTPUT = OUTPUT_DIR / "sequence2function_interpretation_tutorial_executed.ipynb"


def main() -> int:
    if not NOTEBOOK.exists():
        raise FileNotFoundError(NOTEBOOK)
    OUTPUT_DIR.mkdir(exist_ok=True)
    env = os.environ.copy()
    # The notebook selects cuda:0 when only one device is visible.
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    command = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--ExecutePreprocessor.kernel_name=python3",
        "--ExecutePreprocessor.timeout=1800",
        "--output",
        str(OUTPUT),
        str(NOTEBOOK),
    ]
    return subprocess.call(command, cwd=ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())

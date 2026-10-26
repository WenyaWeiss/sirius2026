# Sirius 2026: sequence-to-function interpretation

A self-contained teaching notebook for interpreting Agarwal lentiMPRA
sequence-to-function predictions with ISM, seqlets, HOCOMOCO DNA motifs,
MacroAPE Jaccard similarity, and UPGMA clustering.

The repository contains the exact five example sequences, the real Agarwal
LegNet checkpoint, the bundled LegNet architecture, the clustered HOCOMOCO
v14 motif library, and the MacroAPE runner. No Google Drive folder or external
`mpramnist` checkout is required. An optional
`annotations/precomputed_macroape.json` cache can be generated on the server
for the five target-mode examples; it lets Colab reuse MacroAPE results without
running Java.

## Quick start: local clone

For a private repository, authenticate GitHub first (SSH is recommended):

```bash
git clone git@github.com:WenyaWeiss/sirius2026.git
cd sirius2026
bash scripts/setup.sh
source .venv/bin/activate
jupyter lab sequence_interpretation.ipynb
```

`setup.sh` creates `.venv`, installs `requirements.txt`, and registers a
`Python (sirius2026)` kernel. On a machine with multiple GPUs the notebook uses
the second GPU; with one GPU it uses `cuda:0`; otherwise it runs on CPU.
For a GPU-specific PyTorch build, install the appropriate PyTorch wheel before
running `scripts/setup.sh` or use the existing conda environment.

## Google Colab

The repository is private. Add a fine-grained GitHub token stored in a Colab
Secret named `GITHUB_TOKEN` (Contents: Read-only for this repository), then
open the notebook in Colab. Its first code cell clones the repository into
`/content/sirius2026` and switches to that directory:

```python
from google.colab import userdata
from pathlib import Path
import os
import subprocess

token = userdata.get("GITHUB_TOKEN")
if not token:
    raise RuntimeError("Add GITHUB_TOKEN to Colab Secrets")

askpass = Path("/tmp/github-askpass.sh")
askpass.write_text("""#!/bin/sh
case \"$1\" in
  *Username*) printf '%s\\n' \"$GIT_USERNAME\" ;;
  *)          printf '%s\\n' \"$GIT_PASSWORD\" ;;
esac
""")
askpass.chmod(0o700)
env = os.environ.copy()
env.update({
    "GIT_ASKPASS": str(askpass),
    "GIT_USERNAME": "WenyaWeiss",
    "GIT_PASSWORD": token,
    "GIT_TERMINAL_PROMPT": "0",
})
subprocess.run([
    "git", "clone",
    "https://github.com/WenyaWeiss/sirius2026.git",
    "/content/sirius2026",
], env=env, check=True)

del token
del env["GIT_PASSWORD"]
```

The first cell performs the clone and `%cd`; run the remaining cells from
top to bottom. The next executable notebook cell runs
`!pip install -q -r requirements.txt`,
so the dependencies are installed before the model imports. The image cells
load PNGs from the cloned `pics/` directory with `IPython.display.Image`, so
Colab does not depend on relative Markdown or private GitHub image URLs.

You can open the notebook through the Colab file browser, or use the GitHub
notebook URL after the repository has been authorized:

<https://colab.research.google.com/github/WenyaWeiss/sirius2026/blob/main/sequence_interpretation.ipynb>

The notebook setup cell finds `/content/sirius2026` automatically. It accepts
only a complete repository clone containing `vendor/mpramnist`, `utils.py`,
`sirius_annotation.py`, `models/`, `frameworks/`, and `motifs/`; an older
`/content/drive/MyDrive/sirius_tutorial` folder is deliberately rejected.
Google Drive is only mounted as a fallback when no complete local clone is
found. **MacroAPE is deliberately disabled in Colab:** known sequence+fitness
entries use the cached ISM arrays and MacroAPE motif matches from
`annotations/precomputed_macroape.json`; an unknown sequence+fitness runs ISM
on the selected device (CPU if no GPU is available) and uses tangermeme's
built-in TOMTOM. No Java process is started in Colab.

## What the notebook does

- loads the real Agarwal LegNet checkpoint;
- predicts HepG2, K562, and WTC11 outputs with reverse-complement averaging;
- restricts ISM to the central 200-nt insert;
- computes hypothetical and observed-base ISM tracks;
- extracts seqlets with tangermeme;
- constructs seqlet PFM/PWM matrices using `T=1/15` and `alpha=1/12`;
- matches seqlets to 648 clustered HOCOMOCO v14 DNA motifs with MacroAPE
  (`ape-3.0.6.jar`, `BatchMacroapeVar`, Jaccard similarity) locally/on the server;
- clusters seqlets with average-linkage UPGMA at distance threshold `0.95`;
- in Colab, uses the precomputed MacroAPE cache when available and TOMTOM for
  new sequence+fitness pairs;
- renders ISM panels and motif-reference logo tables.

The notebook contains only lightweight model/orchestration code. Seqlet matrix
construction, MacroAPE execution, TOMTOM fallback, cache loading, and UPGMA
clustering are imported from `sirius_annotation.py`.

`tools/ape-3.0.6.jar` and `tools/BatchMacroapeVar.class` are included in the
private repository. The bundled model architecture is
`vendor/mpramnist/models/MPRALegNet.py`; installing the external `mpramnist`
package is not required.

## Repository layout

```text
sequence_interpretation.ipynb                     # teaching notebook
sirius_annotation.py                              # MacroAPE/TOMTOM/cache/UPGMA engine
utils.py                                          # model, example, and ISM utilities
tutorial_plotting.py                              # all figure/logo rendering
pics/                                              # notebook illustration sources
frameworks/                                       # five selected examples
models/                                            # Agarwal checkpoint
motifs/                                            # clustered HOCOMOCO v14 MEME
tools/                                             # MacroAPE jar + runner (local/server)
annotations/precomputed_macroape.json              # five Colab cache entries
vendor/mpramnist/                                  # bundled LegNet architecture
examples/                                          # checked example figures/table
scripts/setup.sh                                  # create .venv and install deps
```

## Model convention

The checkpoint receives `15 nt flank + 200 nt insert + 15 nt flank` (230 nt
total). Predictions average forward and reverse-complement inputs. Assay
flanks are used by the model but are not interpreted as biological motifs.

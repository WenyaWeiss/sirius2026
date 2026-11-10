# Sirius 2026: sequence-to-function interpretation

A self-contained public workshop notebook for interpreting Agarwal lentiMPRA
sequence-to-function predictions with ISM, seqlets, HOCOMOCO DNA motifs,
MacroAPE Jaccard similarity, consensus-aware PWM alignment, and UPGMA
clustering.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/WenyaWeiss/sirius2026/blob/main/interpretation_workshop.ipynb)

The repository contains the real Agarwal LegNet checkpoint, the bundled LegNet
architecture, the clustered HOCOMOCO v14 motif library, the MacroAPE runner,
and the precomputed data/cache used by `interpretation_workshop.ipynb`. No
Google Drive folder, GitHub token, or external `mpramnist` checkout is required
for the standard workshop run.

## Quick start: local clone

Clone the public repository:

```bash
git clone https://github.com/WenyaWeiss/sirius2026.git
cd sirius2026
bash scripts/setup.sh
source .venv/bin/activate
jupyter lab interpretation_workshop.ipynb
```

`setup.sh` creates `.venv`, installs `requirements.txt`, and registers a
`Python (sirius2026)` kernel. On a machine with multiple GPUs the notebook uses
the second GPU; with one GPU it uses `cuda:0`; otherwise it runs on CPU.
For a GPU-specific PyTorch build, install the appropriate PyTorch wheel before
running `scripts/setup.sh` or use the existing conda environment.

## Google Colab: пошаговый запуск

Репозиторий публичный: `GITHUB_TOKEN`, Colab Secrets и авторизация GitHub для
запуска не требуются.

### Способ 1 — прямая ссылка

1. Откройте
   [`interpretation_workshop.ipynb` в Google Colab](https://colab.research.google.com/github/WenyaWeiss/sirius2026/blob/main/interpretation_workshop.ipynb).
2. Войдите в Google-аккаунт, если Colab попросит это сделать.
3. Если появится предупреждение о notebook из GitHub, выберите
   **Run anyway / Всё равно выполнить**.
4. Выберите **Runtime → Run all** или нажмите `Ctrl+F9`.
5. Дождитесь последовательного выполнения всех ячеек notebook.

### Что notebook делает автоматически

1. Первая code-cell занимается только синхронизацией публичного repository:
   - при первом запуске клонирует ветку `main` в `/content/sirius2026`;
   - при повторном запуске в той же Colab-сессии выполняет
     `git pull --ff-only origin main`.
2. Следующая ячейка находит корень синхронизированного проекта.
3. Отдельная dependency-cell выполняет
   `pip install -q -r requirements.txt`.
4. Следующая ячейка импортирует обычные Python-зависимости.
5. Остальные ячейки загружают model, motifs, annotations и cache из clone и
   выполняют workshop от начала до конца.

Bootstrap не читает `GITHUB_TOKEN`, `google.colab.userdata` или askpass-файлы.
Для повторного чистого запуска можно создать новый Colab runtime и снова
выбрать **Runtime → Run all**.

### Способ 2 — открыть через интерфейс Colab

1. Перейдите на <https://colab.research.google.com/>.
2. Выберите **File → Open notebook**.
3. Откройте вкладку **GitHub**.
4. Вставьте `https://github.com/WenyaWeiss/sirius2026`.
5. Выберите `interpretation_workshop.ipynb`.
6. Запустите **Runtime → Run all**.

### Откуда загружаются данные

Стандартный запуск использует файлы непосредственно из публичного GitHub
clone, в том числе:

- `models/`;
- `frameworks/`;
- `motifs/`;
- `annotations/`;
- `examples/interpretation_workshop_cache_consensus/`;
- `examples/teacher_natural_cache/`;
- `examples/teacher_solutions/`.

Google Drive для стандартного запуска не нужен. Среда `/content` временная;
чтобы сохранить собственную копию notebook с outputs, выберите
**File → Save a copy in Drive**.

### Необязательно: дополнительные данные с Google Drive

Если нужны дополнительные большие или приватные файлы, подключите Drive в
отдельной ячейке после repository-sync cell:

```python
from google.colab import drive
from pathlib import Path

drive.mount('/content/drive')
DRIVE_DATA_ROOT = Path('/content/drive/MyDrive/sirius2026_data')
```

Подключение Drive только делает каталог доступным. Код должен явно читать
дополнительные файлы из `DRIVE_DATA_ROOT`; текущий workshop продолжает
использовать versioned данные из `/content/sirius2026`.

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
- renders ISM panels with an input one-hot sequence logo and true-width motif
  PWM logos aligned to the actual DNA by scanning allowed offsets and both
  forward/reverse-complement orientations; overlapping aligned spans are
  stacked into vertical lanes.

The notebook contains only lightweight model/orchestration code. Seqlet matrix
construction, MacroAPE execution, TOMTOM fallback, cache loading, and UPGMA
clustering are imported from `sirius_annotation.py`.

`tools/ape-3.0.6.jar` and `tools/BatchMacroapeVar.class` are included in the
repository. The bundled model architecture is
`vendor/mpramnist/models/MPRALegNet.py`; installing the external `mpramnist`
package is not required.

## Repository layout

```text
interpretation_workshop.ipynb                     # main public Colab workshop
sequence_interpretation.ipynb                     # foundational teaching notebook
sirius_annotation.py                              # MacroAPE/TOMTOM/cache/UPGMA engine
utils.py                                          # model, ISM, alignment, and rendering utilities
tutorial_plotting.py                              # shared figure/logo helpers
pics/                                             # notebook illustration sources
frameworks/                                       # selected example sequences
models/                                           # Agarwal checkpoint
motifs/                                           # clustered HOCOMOCO v14 MEME
tools/                                            # MacroAPE jar + runner (local/server)
annotations/precomputed_macroape.json             # reusable MacroAPE cache
vendor/mpramnist/                                 # bundled LegNet architecture
examples/interpretation_workshop_cache_consensus/ # workshop figures/annotations
examples/                                         # checked example data and caches
scripts/setup.sh                                  # create .venv and install deps
```

## Model convention

The checkpoint receives `15 nt flank + 200 nt insert + 15 nt flank` (230 nt
total). Predictions average forward and reverse-complement inputs. Assay
flanks are used by the model but are not interpreted as biological motifs.

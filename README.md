# Sirius 2026: интерпретация связи последовательности и функции

Самодостаточный публичный практический ноутбук для интерпретации предсказаний
Agarwal lentiMPRA с помощью ISM, seqlets, ДНК-мотивов HOCOMOCO, сходства
MacroAPE Jaccard, выравнивания PWM по консенсусу и кластеризации UPGMA.

**[Открыть `interpretation_workshop.ipynb` в Google Colab](https://colab.research.google.com/github/WenyaWeiss/sirius2026/blob/main/interpretation_workshop.ipynb)**

Репозиторий содержит исходную контрольную точку Agarwal LegNet, локальную
архитектуру LegNet, библиотеку кластеризованных мотивов HOCOMOCO v14,
программу MacroAPE и предварительно рассчитанные данные для
`interpretation_workshop.ipynb`. Для стандартного запуска не нужны Google
Drive, GitHub token или внешняя установка `mpramnist`.

## Быстрый локальный запуск

Клонируйте публичный репозиторий:

```bash
git clone https://github.com/WenyaWeiss/sirius2026.git
cd sirius2026
bash scripts/setup.sh
source .venv/bin/activate
jupyter lab interpretation_workshop.ipynb
```

Скрипт `setup.sh` создаёт `.venv`, устанавливает зависимости из
`requirements.txt` и регистрирует kernel `Python (sirius2026)`. На машине с
несколькими GPU ноутбук использует вторую GPU; с одной GPU — `cuda:0`; при
отсутствии GPU вычисления выполняются на CPU. Если нужна специальная сборка
PyTorch для конкретной GPU, установите подходящий wheel PyTorch перед запуском
`scripts/setup.sh` или используйте готовое окружение Conda.

## Google Colab: пошаговый запуск

Репозиторий публичный: `GITHUB_TOKEN`, Colab Secrets и авторизация GitHub для
запуска не требуются.

### Способ 1 — прямая ссылка

1. Откройте
   [`interpretation_workshop.ipynb` в Google Colab](https://colab.research.google.com/github/WenyaWeiss/sirius2026/blob/main/interpretation_workshop.ipynb).
2. Войдите в Google-аккаунт, если Colab попросит это сделать.
3. Если появится предупреждение о ноутбуке из GitHub, подтвердите запуск.
4. Выберите **Среда выполнения → Выполнить все** или нажмите `Ctrl+F9`.
5. Дождитесь последовательного выполнения всех ячеек.

### Что выполняется автоматически

1. Первая программная ячейка занимается только синхронизацией публичного
   репозитория:
   - при первом запуске клонирует ветку `main` в `/content/sirius2026`;
   - при повторном запуске в той же Colab-сессии выполняет
     `git pull --ff-only origin main`.
2. Следующая ячейка находит корень синхронизированного проекта.
3. Отдельная ячейка устанавливает зависимости командой
   `pip install -q -r requirements.txt`.
4. Следующая ячейка импортирует обычные Python-зависимости.
5. Остальные ячейки загружают модель, мотивы, аннотации и кеш из clone, после
   чего выполняют практикум от начала до конца.

Стартовый код не читает `GITHUB_TOKEN`, `google.colab.userdata` или
askpass-файлы. Для повторного чистого запуска создайте новую среду выполнения
Colab и снова выберите **Среда выполнения → Выполнить все**.

### Способ 2 — открыть через интерфейс Colab

1. Перейдите на <https://colab.research.google.com/>.
2. Выберите **Файл → Открыть блокнот**.
3. Откройте вкладку **GitHub**.
4. Вставьте `https://github.com/WenyaWeiss/sirius2026`.
5. Выберите `interpretation_workshop.ipynb`.
6. Запустите **Среда выполнения → Выполнить все**.

### Откуда загружаются данные

Стандартный запуск использует файлы непосредственно из публичного clone
репозитория, в том числе:

- `models/`;
- `frameworks/`;
- `motifs/`;
- `annotations/`;
- `examples/interpretation_workshop_cache_consensus/`;
- `examples/teacher_natural_cache/`.

Google Drive для стандартного запуска не нужен. Среда `/content` временная;
чтобы сохранить собственную копию ноутбука вместе с результатами, выберите
**Файл → Сохранить копию на Диске**.

### Необязательно: дополнительные данные с Google Drive

Если нужны дополнительные большие или приватные файлы, подключите Google Drive
в отдельной ячейке после ячейки синхронизации репозитория:

```python
from google.colab import drive

drive.mount('/content/drive')
DRIVE_DATA_ROOT = '/content/drive/MyDrive/sirius2026_data'
```

Подключение Google Drive только делает каталог доступным. Код должен явно
читать дополнительные файлы из `DRIVE_DATA_ROOT`; текущий практикум продолжает
использовать версионированные данные из `/content/sirius2026`.

## Что делает ноутбук

Публичная версия содержит условия исследовательских заданий 1, 2 и 3 без
готовых решений. Задание 5 в публичную версию не включено.

- загружает исходную контрольную точку Agarwal LegNet;
- вычисляет предсказания для HepG2, K562 и WTC11 с усреднением прямой и
  обратно-комплементарной последовательностей;
- ограничивает ISM центральной вставкой длиной 200 нуклеотидов;
- вычисляет hypothetical и observed-base ISM-треки;
- извлекает seqlets с помощью tangermeme;
- строит PFM/PWM-матрицы seqlets с параметрами `T=1/15` и `alpha=1/12`;
- сопоставляет seqlets с 648 кластеризованными ДНК-мотивами HOCOMOCO v14 через
  MacroAPE (`ape-3.0.6.jar`, `BatchMacroapeVar`, сходство Jaccard) локально или
  на сервере;
- кластеризует seqlets методом average-linkage UPGMA с порогом расстояния
  `0.95`;
- в Colab использует предварительно рассчитанный кеш MacroAPE, а для новых пар
  «последовательность + fitness» — встроенный TOMTOM из tangermeme;
- строит ISM-панели с one-hot logo исходной последовательности и PWM-logo
  мотивов истинной ширины;
- выравнивает каждый PWM по реальной ДНК, перебирая допустимые смещения, прямую
  и обратно-комплементарную ориентации, а перекрывающиеся интервалы размещает
  на отдельных вертикальных уровнях.

Ноутбук содержит только лёгкий код модели и управления вычислениями. Построение
матриц seqlets, запуск MacroAPE, резервный TOMTOM, загрузка кеша и кластеризация
UPGMA импортируются из `sirius_annotation.py`.

Файлы `tools/ape-3.0.6.jar` и `tools/BatchMacroapeVar.class` входят в
репозиторий. Архитектура модели находится в
`vendor/mpramnist/models/MPRALegNet.py`; устанавливать внешний пакет
`mpramnist` не требуется.

## Структура репозитория

```text
interpretation_workshop.ipynb                     # основной публичный практикум для Colab
sirius_annotation.py                              # MacroAPE, TOMTOM, кеш и UPGMA
utils.py                                          # модель, ISM, выравнивание и визуализация
tutorial_plotting.py                              # общие функции построения графиков и logo
pics/                                             # исходные иллюстрации ноутбука
frameworks/                                       # выбранные примеры последовательностей
models/                                           # контрольная точка Agarwal
motifs/                                           # HOCOMOCO v14 в формате MEME
tools/                                            # MacroAPE jar и runner
annotations/precomputed_macroape.json             # переиспользуемый кеш MacroAPE
vendor/mpramnist/                                 # локальная архитектура LegNet
examples/interpretation_workshop_cache_consensus/ # фигуры и аннотации практикума
examples/                                         # проверенные данные и кеши примеров
scripts/setup.sh                                  # создание .venv и установка зависимостей
```

## Соглашения модели

Контрольная точка получает последовательность вида
`15 nt flank + 200 nt insert + 15 nt flank`, всего 230 нуклеотидов.
Предсказания усредняются для прямой и обратно-комплементарной
последовательностей. Экспериментальные фланги используются моделью, но не
интерпретируются как биологические мотивы.

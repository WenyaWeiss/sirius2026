# Sirius 2026: интерпретация связи последовательности и функции

Практический ноутбук по интерпретации предсказаний модели Agarwal lentiMPRA
с помощью ISM, seqlets и аннотации мотивов HOCOMOCO.

**[Открыть `interpretation_workshop.ipynb` в Google Colab](https://colab.research.google.com/github/WenyaWeiss/sirius2026/blob/main/interpretation_workshop.ipynb)**

## Запуск в Google Colab

1. Откройте ноутбук по ссылке выше.
2. Если Colab покажет предупреждение о файле из GitHub, подтвердите открытие.
3. Выберите **Среда выполнения → Выполнить все** или нажмите `Ctrl+F9`.

При первом запуске первая программная ячейка клонирует ветку `main` в
`/content/sirius2026`. При повторном запуске в той же сессии она обновляет
каталог командой `git pull --ff-only origin main`. Зависимости устанавливаются
из `requirements.txt` непосредственно в среде Colab.

## Локальный запуск

```bash
git clone https://github.com/WenyaWeiss/sirius2026.git
cd sirius2026
bash scripts/setup.sh
source .venv/bin/activate
jupyter lab interpretation_workshop.ipynb
```

Скрипт `scripts/setup.sh` создаёт `.venv`, устанавливает зависимости из
`requirements.txt` и регистрирует ядро Jupyter `Python (sirius2026)`.

## Структура репозитория

```text
interpretation_workshop.ipynb          # основной ноутбук
requirements.txt                       # Python-зависимости
scripts/setup.sh                       # локальная установка окружения
utils.py                               # модель, ISM, выравнивание и визуализация
sirius_annotation.py                   # аннотация seqlets
teacher_solutions.py                   # загрузка подготовленных данных
tutorial_plotting.py                   # функции построения графиков
models/                                # веса модели
motifs/                                # мотивы HOCOMOCO
annotations/                           # подготовленные аннотации
frameworks/                            # примеры синтетических последовательностей
examples/                              # данные, кеши и фигуры
pics/                                  # иллюстрации ноутбука
tools/                                 # MacroAPE
vendor/mpramnist/                      # архитектура LegNet
```

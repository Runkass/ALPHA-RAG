# ALPHA-RAG

RAG-проект для хакатона Альфа-Банк × МФТИ: по вопросу из `questions.csv` строится ответ на базе корпуса `websites.csv`, результат сохраняется в `submission.csv` (`q_id`, `answer_new`).

## Что в репозитории

- `src/` — ядро пайплайна (retrieval, LLM, постобработка, метрики).
- `scripts/` — утилиты для сборки индекса, генерации submission и локальной оценки.
- `questions.csv`, `websites.csv` — входные данные.
- `sample_submission.csv` — локальный эталон для offline-оценки Recall-L.
- `requirements.txt` — зависимости.

## Быстрый старт (Windows / PowerShell)

```powershell
cd <путь_к_корню_репозитория>
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Заполните API-ключи и нужные параметры в `.env`.

## Сборка индекса

```powershell
.\.venv\Scripts\python.exe scripts\build_index.py
```

## Генерация submission

Полный прогон:

```powershell
.\.venv\Scripts\python.exe scripts\generate_submission.py --refresh --workers 2 --output submission.csv
```

Продолжение после остановки:

```powershell
.\.venv\Scripts\python.exe scripts\generate_submission.py --continue-from submission.csv --workers 2 --output submission.csv
```

Проверка перед загрузкой:

```powershell
.\.venv\Scripts\python.exe scripts\validate_submission.py --file submission.csv
```

## Локальная оценка Recall-L

```powershell
.\.venv\Scripts\python.exe scripts\eval_recall_l.py --pred submission.csv --gold sample_submission.csv
.\.venv\Scripts\python.exe scripts\eval_recall_l.py --pred submission.csv --gold sample_submission.csv --limit 500
```

## Примечания

- `sample_submission.csv` используется только offline.
- Для стабильности на Windows обычно лучше `--workers 1..2`.


import csv
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def test_submission_schema():
    sub = ROOT / "data" / "test_sub.csv"
    sub.parent.mkdir(parents=True, exist_ok=True)
    with sub.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["q_id", "answer_new"])
        w.writerow([1, "Тестовый ответ"])
    df = pd.read_csv(sub)
    assert "answer_new" in df.columns
    assert df.iloc[0]["answer_new"] == "Тестовый ответ"

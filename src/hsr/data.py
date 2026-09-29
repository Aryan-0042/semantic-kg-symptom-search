"""Rows, symptom vocabulary and guideline corpus.

A *row* is one deduplicated record of data/symptom_disease.csv. Its id is its
0-based position in that file, which is the `original_row_index` used by the
v3 split files, so ids are comparable across generations.
"""
from dataclasses import dataclass
import json

import pandas as pd

from . import DATA

SYMPTOM_PREFIX = "Symptom_"


def normalize_symptom(key: str) -> str:
    """Display / matching form of a raw symptom key.

    'dischromic _patches' -> 'dischromic patches'; 'foul_smell_of urine' ->
    'foul smell of urine'. Lower-cased, underscores to spaces, whitespace
    collapsed.
    """
    return " ".join(str(key).replace("_", " ").lower().split())


@dataclass(frozen=True)
class Row:
    row_id: int
    disease: str
    symptoms: tuple  # raw keys, stripped, in dataset column order

    @property
    def symptom_names(self):
        return tuple(normalize_symptom(s) for s in self.symptoms)


def load_rows(path=DATA / "symptom_disease.csv"):
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c.startswith(SYMPTOM_PREFIX)]
    rows = []
    for i, rec in enumerate(df.itertuples(index=False)):
        rec = rec._asdict()
        symptoms = tuple(
            str(rec[c]).strip() for c in cols if pd.notna(rec[c]) and str(rec[c]).strip()
        )
        rows.append(Row(row_id=i, disease=str(rec["Disease"]).strip(), symptoms=symptoms))
    return rows


def symptom_vocabulary(rows):
    """Sorted raw symptom keys across all rows (131 in the current data)."""
    return sorted({s for r in rows for s in r.symptoms})


def diseases(rows):
    return sorted({r.disease for r in rows})


def load_corpus(path=DATA / "guideline_corpus.jsonl"):
    with open(path, encoding="utf-8") as f:
        recs = [json.loads(line) for line in f]
    corpus = pd.DataFrame(recs)
    corpus["disease_tag"] = corpus["disease_tag"].astype(str)
    return corpus

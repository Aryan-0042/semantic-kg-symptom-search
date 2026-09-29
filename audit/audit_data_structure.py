"""
audit_data_structure.py  (RESEARCH_AUDIT.md, sections 4 and 10)

Read-only checks on the dataset and every historical split:
  1. near-duplicate structure of data/symptom_disease.csv
  2. exact and near-duplicate test/dev overlap for v1, v2, v3
  3. test-row reuse across split generations
  4. nested-substitution artefacts in the v3 paraphrased queries

Run from the project root:  venv\\Scripts\\python audit\\audit_data_structure.py
Writes nothing.
"""
from collections import Counter

import numpy as np
import pandas as pd

FULL = pd.read_csv("data/symptom_disease.csv")
SC = [c for c in FULL.columns if c.startswith("Symptom_")]


def sset(row):
    return frozenset(str(v).strip() for v in row[SC] if pd.notna(v) and str(v).strip())


FULL["S"] = FULL.apply(sset, axis=1)


def load(path):
    d = pd.read_csv(path)
    if "Symptom_1" not in d.columns:  # query-format test files -> join back to source rows
        extra = {k: d[k].values for k in d.columns if k not in FULL.columns}
        d = FULL.loc[d.original_row_index].reset_index(drop=True).assign(**extra)
    d["S"] = d.apply(sset, axis=1)
    return d


def dataset_structure():
    print("== 1. dataset structure")
    raw = pd.read_csv("data/raw/dataset.csv")
    print(f"raw rows {len(raw)}; raw rows per disease {sorted(raw.Disease.str.strip().value_counts().unique())}")
    print(f"deduplicated rows {len(FULL)}, diseases {FULL.Disease.nunique()}")
    counts = FULL.Disease.value_counts()
    print(f"rows/disease min {counts.min()} median {counts.median()} max {counts.max()}")
    print(f"symptom sets shared by >1 disease: {(FULL.groupby('S').Disease.nunique() > 1).sum()}")
    near = 0
    for _, g in FULL.groupby("Disease"):
        union = frozenset().union(*g.S)
        near += sum(len(union - s) <= 1 for s in g.S)
    print(f"rows equal to their disease's full symptom union minus <=1 symptom: {near}/{len(FULL)}")
    per_symptom = Counter()
    for _, g in FULL.groupby("Disease"):
        for s in frozenset().union(*g.S):
            per_symptom[s] += 1
    print(f"distinct symptoms {len(per_symptom)}; unique to one disease {sum(v == 1 for v in per_symptom.values())}")


def overlap(name, test, dev):
    keys = set(zip(dev.Disease, dev.S))
    exact = sum((d, s) in keys for d, s in zip(test.Disease, test.S))
    jac, cov = [], []
    for r in test.itertuples():
        g = dev[dev.Disease == r.Disease]
        jac.append(max(len(r.S & s) / len(r.S | s) for s in g.S))
        cov.append(len(r.S & frozenset().union(*g.S)) / len(r.S))
    jac, cov = np.array(jac), np.array(cov)
    print(f"{name}: n_test={len(test)} n_dev={len(dev)} exact_overlap={exact} "
          f"max_jaccard_mean={jac.mean():.3f} (>=0.8: {(jac >= 0.8).sum()}) "
          f"test_symptoms_covered_by_dev_union={cov.mean():.3f}")


def splits():
    print("\n== 2. split overlap")
    t1, d1 = load("data/split_test_final.csv"), load("data/split_dev.csv")
    dt1, dv1 = load("data/split_dev_train.csv"), load("data/split_dev_val.csv")
    t2 = load("data/split_test_final_v2.csv")
    t3, d3 = load("data/split_test_final_v3.csv"), load("data/split_dev_v3.csv")
    dt3, dv3 = load("data/split_dev_train_v3.csv"), load("data/split_dev_val_v3.csv")
    overlap("v1 test vs v1 dev", t1, d1)
    overlap("v2 test vs v1 dev (index pool of the v2 run)", t2, d1)
    overlap("v3 test vs v3 dev", t3, d3)
    overlap("v1 dev-val vs v1 dev-train", dv1, dt1)
    overlap("v3 dev-val vs v3 dev-train", dv3, dt3)

    print("\n== 3. test-row reuse across generations")
    i3 = set(t3.original_row_index)
    print(f"v3 test rows also in v2 test: {len(i3 & set(t2.original_row_index))}")
    k1 = set(zip(t1.Disease, t1.S))
    print(f"v3 test rows also in v1 test: {sum((d, s) in k1 for d, s in zip(t3.Disease, t3.S))}")


def paraphrase_artefacts():
    print("\n== 4. paraphrase artefacts (v3 test)")
    t = pd.read_csv("data/split_test_final_v3.csv")
    p = t[t.paraphrased]
    bad = p[p.query_text.str.contains("won't quiting|an my skin is really itchy", regex=True)]
    print(f"paraphrased rows {len(p)}; rows with nested-substitution artefacts {len(bad)}")
    for q in bad.query_text:
        print("   ", q)
    kept = []
    for r in p.itertuples():
        true = [x.strip() for x in r.canonical_query_text.replace("The patient reports ", "").rstrip(".").split(", ")]
        kept.append(np.mean([s in r.query_text.lower() for s in true]))
    print(f"mean fraction of canonical symptom strings still literally present: {np.mean(kept):.2f}")


if __name__ == "__main__":
    dataset_structure()
    splits()
    paraphrase_artefacts()

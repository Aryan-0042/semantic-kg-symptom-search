"""Overlap of the v4 final-test partition with every historical evaluation set.

    venv\\Scripts\\python audit\\audit_test_overlap.py

Maps every historical query set to row ids of data/symptom_disease.csv
(the 304 deduplicated rows), verifies each mapping, and reports how many of
the 181 v4 test rows (data/protocol/partition.csv) were previously evaluated.
No retrieval is run.

Historical sets (see RESEARCH_AUDIT.md section 4 and the scripts named below):

  retrieval, held-out test
    v1_test     data/split_test_final.csv       retrieval_eval_final(_L2).py; content only, no row id
    v2_test     data/split_test_final_v2.csv    retrieval_eval_final_v2.py, latency_benchmark.py
    v3_test     data/split_test_final_v3.csv    retrieval_eval_final_v3.py, master_results_verification_v3.py,
                                                quick_recency_variance_check.py, audit/audit_diagnostics.py
  retrieval, tuning validation (results observed during tuning)
    v1_devval   data/split_dev_val.csv          tune_crfs_weights.py, tune_rrf_shortlist.py; content only
    v3_devval   data/split_dev_val_v3.csv       tune_crfs_weights_v3.py, tune_rrf_shortlist_v3.py, rrf_*_diagnostic_v3.py
  retrieval, v0 (reconstructed from src/retrieval_eval_legacy.py; which variant
  produced results/*_OLD_100pct.csv is not recorded, so both are counted)
    v0_7015_test    stratified 70/15/15 split, seed 42 (= prepare_dataset.py split_test.csv)
    v0_one_per_dis  one row per disease, RandomState(42)
  other tasks (no disease-retrieval outcome per row; reported separately)
    triage_cv   5-fold CV over all 304 rows, urgency label, aggregate accuracy only
    igft        900 draws with replacement, unseeded, no row ids saved
    safety      150 rows sampled (random_state 42) as benign texts for a safety set
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "audit" / "outputs"
sys.path.insert(0, str(ROOT / "src"))
from hsr.data import load_rows, normalize_symptom  # noqa: E402

ROWS = load_rows()
RAW = pd.read_csv(DATA / "symptom_disease.csv")
SYM_COLS = [c for c in RAW.columns if c.startswith("Symptom_")]


def row_key(disease, symptoms):
    return (str(disease).strip(), tuple(str(s).strip() for s in symptoms if pd.notna(s) and str(s).strip()))


KEY_TO_ID = {row_key(r.disease, r.symptoms): r.row_id for r in ROWS}
assert len(KEY_TO_ID) == len(ROWS), "row content keys must be unique"


def ids_by_content(path):
    df = pd.read_csv(path)
    ids = []
    for rec in df.itertuples(index=False):
        rec = rec._asdict()
        key = row_key(rec["Disease"], [rec[c] for c in SYM_COLS])
        if key not in KEY_TO_ID:
            raise ValueError(f"{path.name}: row not found among the 304 rows: {key}")
        ids.append(KEY_TO_ID[key])
    return ids


def ids_by_index(path):
    """Row ids from original_row_index, checked against disease and query text."""
    df = pd.read_csv(path)
    ids = df.original_row_index.astype(int).tolist()
    for rec, rid in zip(df.itertuples(index=False), ids):
        row = ROWS[rid]
        assert row.disease == str(rec.Disease).strip(), (path.name, rid)
        text = " ".join(str(rec.canonical_query_text).lower().split())  # historical text kept double spaces
        missing = [s for s in row.symptom_names if s not in text]
        assert not missing, f"{path.name}: row {rid} symptoms missing from canonical_query_text: {missing}"
    return ids


def v0_sets():
    df = pd.read_csv(DATA / "symptom_disease.csv")
    _, temp = train_test_split(df, test_size=0.30, random_state=42, stratify=df["Disease"])
    _, test = train_test_split(temp, test_size=0.50, random_state=42, stratify=None)
    rng = np.random.RandomState(42)
    one = [int(rng.choice(g.index.to_numpy())) for _, g in df.groupby("Disease", sort=True)]
    return [int(i) for i in test.index], one


def safety_ids():
    return [int(i) for i in pd.read_csv(DATA / "symptom_disease.csv").sample(150, random_state=42).index]


def main():
    part = pd.read_csv(DATA / "protocol" / "partition.csv")
    split = dict(zip(part.row_id, part.split))
    test_ids = set(part.row_id[part.split == "test"])
    dev_ids = set(part.row_id[part.split == "dev"])
    disease_of = {r.row_id: r.disease for r in ROWS}

    v0_test, v0_one = v0_sets()
    sets = {  # name -> (category, query row ids in query order)
        "v1_test": ("retrieval test", ids_by_content(DATA / "split_test_final.csv")),
        "v2_test": ("retrieval test", ids_by_index(DATA / "split_test_final_v2.csv")),
        "v3_test": ("retrieval test", ids_by_index(DATA / "split_test_final_v3.csv")),
        "v1_devval": ("retrieval tuning validation", ids_by_content(DATA / "split_dev_val.csv")),
        "v3_devval": ("retrieval tuning validation", ids_by_content(DATA / "split_dev_val_v3.csv")),
        "v0_7015_test": ("retrieval v0 (reconstructed)", v0_test),
        "v0_one_per_disease": ("retrieval v0 (reconstructed)", v0_one),
        "safety_benign": ("other task: safety texts", safety_ids()),
        "triage_cv": ("other task: urgency CV", [r.row_id for r in ROWS]),
    }
    # Sanity checks against stored result files (query counts per method).
    assert len(sets["v1_test"][1]) == 82 and len(sets["v2_test"][1]) == 82 and len(sets["v3_test"][1]) == 123
    assert len(sets["v1_devval"][1]) == 41 and len(sets["v3_devval"][1]) == 33
    v3_dev = set(pd.read_csv(DATA / "split_dev_v3.csv").original_row_index)
    assert set(sets["v3_test"][1]).isdisjoint(v3_dev) and len(v3_dev | set(sets["v3_test"][1])) == 304

    by_set = []
    for name, (cat, ids) in sets.items():
        u = set(ids)
        by_set.append({"set": name, "category": cat, "queries": len(ids), "unique_rows": len(u),
                       "diseases": len({disease_of[i] for i in u}),
                       "rows_in_v4_test": len(u & test_ids), "rows_in_v4_dev": len(u & dev_ids),
                       "share_of_v4_test_touched": round(len(u & test_ids) / len(test_ids), 3)})
    by_set = pd.DataFrame(by_set)

    tiers = {
        "A: historical test sets (v1, v2, v3)": ["v1_test", "v2_test", "v3_test"],
        "B: A + tuning validation sets (v1, v3)": ["v1_test", "v2_test", "v3_test", "v1_devval", "v3_devval"],
        "C: B + v0 reconstructed (both variants)": ["v1_test", "v2_test", "v3_test", "v1_devval", "v3_devval",
                                                     "v0_7015_test", "v0_one_per_disease"],
    }
    tier_rows = []
    for tname, names in tiers.items():
        u = set().union(*(set(sets[n][1]) for n in names))
        tier_rows.append({"tier": tname, "union_rows": len(u), "union_in_v4_dev": len(u & dev_ids),
                          "union_in_v4_test": len(u & test_ids), "v4_test_untouched": len(test_ids - u)})
    tiers_df = pd.DataFrame(tier_rows)

    retrieval_sets = [n for n, (c, _) in sets.items() if c.startswith("retrieval")]
    per_row = pd.DataFrame({"row_id": sorted(test_ids)})
    per_row["disease"] = per_row.row_id.map(disease_of)
    per_row["n_symptoms"] = per_row.row_id.map(lambda i: len(ROWS[i].symptoms))
    for n in retrieval_sets:
        per_row[n] = per_row.row_id.isin(set(sets[n][1])).astype(int)
    per_row["n_historical_retrieval_sets"] = per_row[retrieval_sets].sum(1)
    per_row["v3_role"] = per_row.row_id.map(lambda i: "test" if i in set(sets["v3_test"][1]) else "dev(index/tuning)")

    union_c = set().union(*(set(sets[n][1]) for n in retrieval_sets))
    union_a = set().union(*(set(sets[n][1]) for n in tiers["A: historical test sets (v1, v2, v3)"]))
    by_dis = []
    for d in sorted({r.disease for r in ROWS}):
        rows_d = {r.row_id for r in ROWS if r.disease == d}
        t_d = rows_d & test_ids
        rec = {"disease": d, "rows": len(rows_d), "v4_dev": len(rows_d & dev_ids), "v4_test": len(t_d)}
        for n in retrieval_sets:
            rec[f"{n}_rows(test)"] = f"{len(rows_d & set(sets[n][1]))}({len(t_d & set(sets[n][1]))})"
        rec["v4_test_in_test_sets_A"] = len(t_d & union_a)
        rec["v4_test_in_any_retrieval_C"] = len(t_d & union_c)
        rec["v4_test_untouched_C"] = len(t_d - union_c)
        by_dis.append(rec)
    by_dis = pd.DataFrame(by_dis)

    OUT.mkdir(parents=True, exist_ok=True)
    by_set.to_csv(OUT / "test_overlap_by_set.csv", index=False, lineterminator="\n")
    tiers_df.to_csv(OUT / "test_overlap_tiers.csv", index=False, lineterminator="\n")
    by_dis.to_csv(OUT / "test_overlap_by_disease.csv", index=False, lineterminator="\n")
    per_row.to_csv(OUT / "test_overlap_by_row.csv", index=False, lineterminator="\n")

    pd.set_option("display.width", 250)
    print(f"sklearn {sklearn.__version__} (v0 reconstruction depends on train_test_split behaviour)")
    print(f"v4 partition: {len(dev_ids)} dev rows, {len(test_ids)} test rows\n")
    print("Per historical set\n" + by_set.to_string(index=False) + "\n")
    print("Unions\n" + tiers_df.to_string(index=False) + "\n")
    pairs = [(a, b, len(set(sets[a][1]) & set(sets[b][1]))) for i, a in enumerate(retrieval_sets) for b in retrieval_sets[i + 1:]]
    print("Pairwise row overlap between historical retrieval sets")
    print(pd.DataFrame(pairs, columns=["set_a", "set_b", "shared_rows"]).to_string(index=False) + "\n")
    print("v4 test rows by number of historical retrieval sets containing them")
    print(per_row.n_historical_retrieval_sets.value_counts().sort_index().to_string() + "\n")
    print("v4 test rows by v3 role: " + per_row.v3_role.value_counts().to_dict().__repr__() + "\n")
    print("Per disease\n" + by_dis[["disease", "rows", "v4_dev", "v4_test", "v4_test_in_test_sets_A",
                                   "v4_test_in_any_retrieval_C", "v4_test_untouched_C"]].to_string(index=False))
    print(f"\ndiseases with every v4 test row historically evaluated (C): "
          f"{int((by_dis.v4_test_untouched_C == 0).sum())} / {len(by_dis)}")
    print(f"diseases with no untouched v4 test row under A: {int((by_dis.v4_test - by_dis.v4_test_in_test_sets_A == 0).sum())}")


if __name__ == "__main__":
    main()

"""
make_frozen_test_split_v3.py

Builds the FROZEN, asymmetric v3 confirmatory test set: for each of the
41 diseases, draws 3 rows from the FULL original dataset (not from any
prior dev/test split) with a NEW seed, distinct from both the v1
(seed=42) and v2 (seed=99) splits. One drawn row is assigned canonical,
two are independently assigned to the paraphrased treatment.

123 rows total = 41 canonical + 82 paraphrased. This is NOT "50% more
independent samples" than the v2 82-query set -- it is broader
disease-balanced coverage plus a strengthened two-instance paraphrase
stress test per disease, and should be described that way in the paper.

Run this ONCE. After this script has been run, data/split_test_final_v3.csv
must not be inspected for tuning purposes, regenerated with a different
seed, or rebalanced after seeing any downstream result.

A companion file, data/split_dev_v3.csv, is written as the EXACT
complement of the test rows (a pure set-difference on original row
index against the full dataset) -- this guarantees disjointness by
construction, which is the actual fix for the audited dev/test overlap
problem. It does not rely on probability or a second differently-seeded
split the way the old split_dev.csv did.
"""
import pandas as pd
import numpy as np

from paraphrase_dictionary import paraphrase_query_text

SEED = 157  # distinct from v1 (42) and v2 (99)
N_PER_DISEASE = 3
IN_PATH = "data/symptom_disease.csv"
TEST_OUT = "data/split_test_final_v3.csv"
DEV_OUT = "data/split_dev_v3.csv"
COVERAGE_OUT = "results/paraphrase_coverage_v3.csv"


def get_symptom_cols(df):
    return [c for c in df.columns if c.lower().startswith("symptom_") and c.lower() != "symptom_set"]


def build_query_text(row, symptom_cols):
    symptoms = []
    for col in symptom_cols:
        v = row[col]
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s:
            symptoms.append(s.replace("_", " "))
    return "The patient reports " + ", ".join(symptoms) + "."


def main():
    df = pd.read_csv(IN_PATH).reset_index(drop=True)
    rng = np.random.RandomState(SEED)
    symptom_cols = get_symptom_cols(df)

    too_small = []
    for disease, group in df.groupby("Disease", sort=True):
        if len(group) < N_PER_DISEASE:
            too_small.append((disease, len(group)))
    if too_small:
        raise ValueError(
            f"These diseases have fewer than {N_PER_DISEASE} rows and cannot support "
            f"the 1-canonical + 2-paraphrased draw: {too_small}"
        )

    test_rows = []
    coverage_rows = []

    for disease, group in df.groupby("Disease", sort=True):
        idx = group.index.to_numpy()
        chosen = rng.choice(idx, size=N_PER_DISEASE, replace=False)
        # slot 0 = canonical, slots 1 and 2 = independently paraphrased
        slots = ["canonical", "paraphrased_1", "paraphrased_2"]
        # shuffle which of the 3 drawn rows gets which slot, still deterministic under SEED
        rng.shuffle(chosen)

        for row_idx, slot in zip(chosen, slots):
            row = df.loc[row_idx]
            canonical_text = build_query_text(row, symptom_cols)
            is_paraphrased = slot != "canonical"

            if is_paraphrased:
                final_text, applied_terms = paraphrase_query_text(canonical_text, rng, apply_probability=1.0)
            else:
                final_text, applied_terms = canonical_text, []

            test_rows.append({
                "Disease": row["Disease"],
                "original_row_index": int(row_idx),
                "paraphrase_slot": slot,
                "query_text": final_text,
                "canonical_query_text": canonical_text,
                "paraphrased": is_paraphrased,
                "paraphrased_terms": "|".join(applied_terms),
            })

            if is_paraphrased:
                coverage_rows.append({
                    "Disease": row["Disease"],
                    "original_row_index": int(row_idx),
                    "n_terms_substituted": len(applied_terms),
                    "substitution_occurred": len(applied_terms) > 0,
                })

    out_df = pd.DataFrame(test_rows)
    dev_df = df.drop(index=out_df["original_row_index"].tolist()).copy()
    dev_df.insert(0, "original_row_index", dev_df.index)
    dev_df = dev_df.reset_index(drop=True)

    # ---- Hard integrity checks, fail loudly rather than silently ----
    assert out_df["Disease"].nunique() == df["Disease"].nunique(), \
        "Not every disease is represented in the frozen test set."
    assert (out_df.groupby("Disease").size() == N_PER_DISEASE).all(), \
        "Not every disease has exactly 3 test rows."
    assert (out_df.groupby("Disease")["paraphrased"].sum() == 2).all(), \
        "Not every disease has exactly 2 paraphrased rows."
    assert (out_df.groupby("Disease")["paraphrased"].apply(lambda s: (~s).sum()) == 1).all(), \
        "Not every disease has exactly 1 canonical row."
    assert set(out_df["original_row_index"]).isdisjoint(set(dev_df["original_row_index"])), \
        "Test and development rows overlap -- this must never happen."
    assert len(out_df) + len(dev_df) == len(df), \
        "Test + dev row counts don't sum to the full dataset -- a row was lost or duplicated."

    import os
    os.makedirs("results", exist_ok=True)
    out_df.to_csv(TEST_OUT, index=False)
    dev_df.to_csv(DEV_OUT, index=False)

    coverage_df = pd.DataFrame(coverage_rows)
    coverage_df.to_csv(COVERAGE_OUT, index=False)

    n_no_substitution = int((~coverage_df["substitution_occurred"]).sum())

    print("=" * 60)
    print("FROZEN v3 TEST SPLIT CREATED (123 queries)")
    print("=" * 60)
    print(f"Total rows in source dataset : {len(df)}")
    print(f"Diseases                     : {df['Disease'].nunique()}")
    print(f"Frozen test rows             : {len(out_df)}  (41 canonical + 82 paraphrased) -> {TEST_OUT}")
    print(f"Development rows             : {len(dev_df)}  (exact complement, guaranteed disjoint) -> {DEV_OUT}")
    print(f"Seed                         : {SEED}")
    print(f"\nParaphrase coverage check    : {len(coverage_df) - n_no_substitution}/{len(coverage_df)} "
          f"paraphrased rows had >=1 term substituted -> {COVERAGE_OUT}")
    if n_no_substitution > 0:
        print(f"  WARNING: {n_no_substitution} paraphrased row(s) had ZERO dictionary terms match.")
        print("  These rows are functionally canonical-phrased despite being labeled 'paraphrased'.")
        print("  Review results/paraphrase_coverage_v3.csv before treating the split as final.")
    else:
        print("  All paraphrased rows received at least one substitution. Good.")

    print("\nDo not modify, regenerate, or rebalance this split after this point.")
    print("It is now frozen, and no result has been computed against it yet.")


if __name__ == "__main__":
    main()

"""
make_new_frozen_test_split.py

Draws a FRESH, independent test set for the Option-A confirmatory
evaluation. Per the discussion: drawn from the FULL original
data/symptom_disease.csv (304 rows), with a NEW seed, disclosed as such
in the paper. Row overlap with the original 82-query test set is
possible and explicitly not a problem -- what matters is that nothing
in THIS test set touched THIS round of dev-train/dev-val tuning
(tune_rrf_shortlist.py), which it does not, since that script never
reads this file or the original symptom_disease.csv directly.

PRE-REGISTERED protocol (decided now, before any RRF+KG+Recency result
exists, exactly as agreed):
  - 82 rows total, 2 per disease (same size/shape as before, for a
    like-for-like comparison)
  - Exactly half of the 82 queries (41, one per disease) are rewritten
    using the pre-registered paraphrase dictionary; the other half keep
    the original canonical-vocabulary phrasing. Which of a disease's 2
    rows gets paraphrased is decided by a fixed seeded coin flip, not
    hand-picked.
  - This is fully disclosed and saved to disk with a `paraphrased`
    column and a `paraphrased_terms` column recording exactly what was
    substituted, so the eventual paper can report accuracy split by
    subset rather than only in aggregate.

SEED = 99, deliberately distinct from both the original test-split seed
(42) and the dev-train/dev-val seed (43).
"""
import pandas as pd
import numpy as np

from paraphrase_dictionary import paraphrase_query_text

SEED = 99
N_TEST_PER_DISEASE = 2
IN_PATH = "data/symptom_disease.csv"
OUT_PATH = "data/split_test_final_v2.csv"


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
    symptom_cols = [c for c in df.columns if c.lower().startswith("symptom_") and c.lower() != "symptom_set"]

    test_rows = []
    too_small = []
    for disease, group in df.groupby("Disease", sort=True):
        idx = group.index.to_numpy()
        if len(idx) < N_TEST_PER_DISEASE:
            too_small.append((disease, len(idx)))
            continue
        chosen = rng.choice(idx, size=N_TEST_PER_DISEASE, replace=False)
        # Fixed seeded coin flip: exactly one of the two chosen rows
        # per disease is marked for paraphrasing.
        paraphrase_flags = [False, False]
        paraphrase_flags[rng.randint(0, 2)] = True

        for row_idx, do_paraphrase in zip(chosen, paraphrase_flags):
            row = df.loc[row_idx]
            query_text = build_query_text(row, symptom_cols)
            if do_paraphrase:
                final_text, applied_terms = paraphrase_query_text(query_text, rng, apply_probability=1.0)
            else:
                final_text, applied_terms = query_text, []

            test_rows.append({
                "Disease": row["Disease"],
                "original_row_index": int(row_idx),
                "query_text": final_text,
                "canonical_query_text": query_text,
                "paraphrased": do_paraphrase,
                "paraphrased_terms": "|".join(applied_terms),
            })

    if too_small:
        raise ValueError(f"Diseases with too few rows for {N_TEST_PER_DISEASE} test samples: {too_small}")

    out_df = pd.DataFrame(test_rows)

    assert out_df["Disease"].nunique() == df["Disease"].nunique()
    assert (out_df["Disease"].value_counts() == N_TEST_PER_DISEASE).all()
    assert out_df["paraphrased"].sum() == out_df["Disease"].nunique(), \
        "Expected exactly one paraphrased row per disease."

    out_df.to_csv(OUT_PATH, index=False)

    print("=" * 60)
    print("NEW CONFIRMATORY TEST SET CREATED")
    print("=" * 60)
    print(f"Total test queries       : {len(out_df)}")
    print(f"Diseases                 : {out_df['Disease'].nunique()}")
    print(f"Paraphrased subset       : {out_df['paraphrased'].sum()} "
          f"({100*out_df['paraphrased'].mean():.1f}%)")
    print(f"Canonical subset         : {(~out_df['paraphrased']).sum()}")
    print(f"Seed                     : {SEED}")

    print(f"\nSaved to {OUT_PATH}")
    print("Do not modify or regenerate this file after this point. It is now frozen,")
    print("and no result from any method has been computed against it yet.")


if __name__ == "__main__":
    main()

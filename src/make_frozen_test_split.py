"""
make_frozen_test_split.py

Creates the FROZEN final test set: exactly 2 samples per disease (82 rows
for 41 diseases), deterministic under SEED. The remaining ~222 rows become
the development set. Run this ONCE. After this script has been run, the
resulting data/split_test_final.csv must not be inspected again until the
single final evaluation run (retrieval_eval_final.py) -- no peeking, no
re-running this script with a different seed after seeing bad results.

Every disease in this dataset has >=5 rows (per the distribution you
provided), so removing 2 per disease always leaves >=3 for development.
The script asserts this explicitly and fails loudly if any disease has
fewer than 3 rows remaining after the split, rather than silently
producing an unusable development set for that class.
"""
import pandas as pd
import numpy as np

SEED = 42
N_TEST_PER_DISEASE = 2
IN_PATH = "data/symptom_disease.csv"
TEST_OUT = "data/split_test_final.csv"
DEV_OUT = "data/split_dev.csv"


def main():
    df = pd.read_csv(IN_PATH)
    df = df.reset_index(drop=True)

    rng = np.random.RandomState(SEED)

    test_indices = []
    too_small = []
    for disease, group in df.groupby("Disease", sort=True):
        idx = group.index.to_numpy()
        if len(idx) < N_TEST_PER_DISEASE + 1:
            too_small.append((disease, len(idx)))
            continue
        chosen = rng.choice(idx, size=N_TEST_PER_DISEASE, replace=False)
        test_indices.extend(chosen.tolist())

    if too_small:
        raise ValueError(
            f"These diseases have fewer than {N_TEST_PER_DISEASE + 1} rows and "
            f"cannot safely contribute to both test and development sets: {too_small}. "
            f"Resolve this before freezing the split (e.g. reduce N_TEST_PER_DISEASE, "
            f"or collect more rows for these classes)."
        )

    test_df = df.loc[sorted(test_indices)].copy()
    dev_df = df.drop(index=test_indices).copy()

    # Hard guarantees -- fail loudly rather than silently producing a
    # skewed or incomplete split.
    assert test_df["Disease"].nunique() == df["Disease"].nunique(), \
        "Not every disease is represented in the frozen test set."
    assert (test_df["Disease"].value_counts() == N_TEST_PER_DISEASE).all(), \
        "Not every disease has exactly N_TEST_PER_DISEASE test rows."
    assert set(test_df.index).isdisjoint(set(dev_df.index)), \
        "Test and development rows overlap -- this must never happen."

    test_df.to_csv(TEST_OUT, index=False)
    dev_df.to_csv(DEV_OUT, index=False)

    print("=" * 60)
    print("FROZEN TEST SPLIT CREATED")
    print("=" * 60)
    print(f"Total rows           : {len(df)}")
    print(f"Diseases              : {df['Disease'].nunique()}")
    print(f"Frozen test rows      : {len(test_df)}  -> {TEST_OUT}")
    print(f"Development rows      : {len(dev_df)}  -> {DEV_OUT}")
    print(f"Seed                  : {SEED}")
    print("\nDev rows per disease (min/max/mean):",
          dev_df["Disease"].value_counts().min(),
          dev_df["Disease"].value_counts().max(),
          round(dev_df["Disease"].value_counts().mean(), 2))
    print("\nDo not modify or regenerate this file with a different seed")
    print("after this point. It is now frozen.")


if __name__ == "__main__":
    main()

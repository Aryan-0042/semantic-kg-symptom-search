"""
make_dev_train_val_split.py

Splits data/split_dev.csv (~222 rows, produced by make_frozen_test_split.py)
into dev-train and dev-val. dev-val is the ONLY data allowed to influence
CRFS weight selection or KG parameter choices. dev-train is what
build_retrieval_text() and build_kg() are built from during tuning.

sklearn's stratified split is avoided here on purpose: after removing 2
test rows per disease, several classes are down to 3-4 rows, and a
20% stratified split on classes that small either errors out (as you
saw before) or silently produces 0-sample val folds for the smallest
classes. Instead we use the same deterministic "peel off 1 sample per
class where possible" approach already validated for the test split,
which guarantees every disease that can appear in val does appear in
val, and never touches a class down to zero dev-train rows.
"""
import pandas as pd
import numpy as np

SEED = 43  # deliberately different from the test-split seed (42)
IN_PATH = "data/split_dev.csv"
TRAIN_OUT = "data/split_dev_train.csv"
VAL_OUT = "data/split_dev_val.csv"

MIN_TRAIN_ROWS_PER_DISEASE = 2  # never let dev-train drop below this for any class


def main():
    df = pd.read_csv(IN_PATH).reset_index(drop=True)
    rng = np.random.RandomState(SEED)

    val_indices = []
    skipped = []
    for disease, group in df.groupby("Disease", sort=True):
        idx = group.index.to_numpy()
        if len(idx) - 1 < MIN_TRAIN_ROWS_PER_DISEASE:
            skipped.append((disease, len(idx)))
            continue
        chosen = rng.choice(idx, size=1, replace=False)
        val_indices.extend(chosen.tolist())

    val_df = df.loc[sorted(val_indices)].copy()
    train_df = df.drop(index=val_indices).copy()

    assert set(val_df.index).isdisjoint(set(train_df.index))

    train_df.to_csv(TRAIN_OUT, index=False)
    val_df.to_csv(VAL_OUT, index=False)

    print("=" * 60)
    print("DEV-TRAIN / DEV-VAL SPLIT CREATED")
    print("=" * 60)
    print(f"Development rows total : {len(df)}")
    print(f"dev-train rows         : {len(train_df)}  -> {TRAIN_OUT}")
    print(f"dev-val rows           : {len(val_df)}  -> {VAL_OUT}")
    print(f"Diseases in dev-val     : {val_df['Disease'].nunique()} / {df['Disease'].nunique()}")
    if skipped:
        print(f"\nNOTE: {len(skipped)} disease(s) had too few dev rows to contribute a "
              f"val sample without dropping dev-train below {MIN_TRAIN_ROWS_PER_DISEASE}: {skipped}")
        print("These diseases are only evaluated for weight-tuning purposes via the")
        print("diseases that ARE in dev-val -- they still fully participate in the")
        print("frozen final test set, this only affects which classes inform tuning.")


if __name__ == "__main__":
    main()

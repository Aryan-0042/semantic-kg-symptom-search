"""
make_dev_train_val_split_v3.py

Splits data/split_dev_v3.csv (181 rows, the exact complement of the
frozen 123-query test set) into dev-train and dev-val for hyperparameter
tuning. Same mechanism as the v2 split: a disease contributes exactly
one row to dev-val only if doing so leaves at least
MIN_TRAIN_ROWS_PER_DISEASE rows behind in dev-train; otherwise the
disease is skipped from dev-val entirely and both its remaining rows
stay in dev-train.

CHANGE FROM v2: the skip decision is now written to an explicit,
citable output file (results/dev_val_coverage_v3.csv) rather than only
being visible in console output. Given 3 rows/disease were removed for
the frozen test set (vs. 2 previously), more diseases are expected to
hit this guard than under v2 -- this must be disclosed in the paper,
not just left in a log.
"""
import pandas as pd
import numpy as np

SEED = 211  # distinct from the v2 dev-train/val seed (43) and the v3 test-split seed (157)
IN_PATH = "data/split_dev_v3.csv"
TRAIN_OUT = "data/split_dev_train_v3.csv"
VAL_OUT = "data/split_dev_val_v3.csv"
COVERAGE_OUT = "results/dev_val_coverage_v3.csv"

MIN_TRAIN_ROWS_PER_DISEASE = 2  # unchanged guard threshold from v2


def main():
    df = pd.read_csv(IN_PATH).reset_index(drop=True)
    rng = np.random.RandomState(SEED)

    val_indices = []
    coverage_rows = []

    for disease, group in df.groupby("Disease", sort=True):
        idx = group.index.to_numpy()
        dev_pool_count = len(idx)
        would_leave_in_train = dev_pool_count - 1

        if would_leave_in_train < MIN_TRAIN_ROWS_PER_DISEASE:
            coverage_rows.append({
                "Disease": disease,
                "dev_pool_rows": dev_pool_count,
                "represented_in_dev_val": False,
                "dev_train_rows_final": dev_pool_count,
                "reason": f"dev_pool_rows - 1 = {would_leave_in_train} < "
                          f"MIN_TRAIN_ROWS_PER_DISEASE ({MIN_TRAIN_ROWS_PER_DISEASE})",
            })
            continue

        chosen = rng.choice(idx, size=1, replace=False)
        val_indices.extend(chosen.tolist())
        coverage_rows.append({
            "Disease": disease,
            "dev_pool_rows": dev_pool_count,
            "represented_in_dev_val": True,
            "dev_train_rows_final": dev_pool_count - 1,
            "reason": "",
        })

    val_df = df.loc[sorted(val_indices)].copy()
    train_df = df.drop(index=val_indices).copy()

    assert set(val_df.index).isdisjoint(set(train_df.index))
    assert len(val_df) + len(train_df) == len(df)

    train_df.to_csv(TRAIN_OUT, index=False)
    val_df.to_csv(VAL_OUT, index=False)

    coverage_df = pd.DataFrame(coverage_rows).sort_values("Disease")
    import os
    os.makedirs("results", exist_ok=True)
    coverage_df.to_csv(COVERAGE_OUT, index=False)

    n_represented = int(coverage_df["represented_in_dev_val"].sum())
    n_excluded = len(coverage_df) - n_represented
    min_devtrain = int(coverage_df["dev_train_rows_final"].min())

    print("=" * 60)
    print("DEV-TRAIN / DEV-VAL SPLIT CREATED (v3, corrected pool)")
    print("=" * 60)
    print(f"Development rows total       : {len(df)}")
    print(f"dev-train rows               : {len(train_df)}  -> {TRAIN_OUT}")
    print(f"dev-val rows                 : {len(val_df)}  -> {VAL_OUT}")
    print(f"Diseases represented in dev-val : {n_represented} / {len(coverage_df)}")
    print(f"Diseases EXCLUDED from dev-val  : {n_excluded} / {len(coverage_df)}")
    print(f"Minimum dev-train rows for any disease : {min_devtrain}")
    print(f"\nFull per-disease coverage table -> {COVERAGE_OUT}")

    if n_excluded > 0:
        excluded_list = coverage_df[~coverage_df["represented_in_dev_val"]]["Disease"].tolist()
        print(f"\nExcluded diseases (no dev-val representation, both rows kept in dev-train):")
        for d in excluded_list:
            print(f"  - {d}")
        print("\nThese diseases contribute zero signal to the CRFS-weight and RRF-shortlist")
        print("hyperparameter search objective. This must be disclosed in the paper's")
        print("methodology section, not left implicit.")


if __name__ == "__main__":
    main()

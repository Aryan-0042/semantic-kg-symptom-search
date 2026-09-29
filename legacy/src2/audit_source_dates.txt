"""
audit_source_dates.py

Reports how many of the corpus documents have a real, extracted
source_date vs. null (recall: build_guideline_corpus.py deliberately
never fakes today's date -- null means "no trustworthy date found").

This determines whether the recency term in CRFS is actually
discriminative or whether it collapses to a near-constant 0.5 for
almost every document (which would make it a mathematical no-op inside
an argsort ranking, regardless of its assigned weight). The paper must
report whichever of these is true -- this script tells you which one it is.
"""
import pandas as pd

IN_PATH = "data/guideline_corpus.jsonl"
OUT_PATH = "results/source_date_audit.csv"


def main():
    df = pd.read_json(IN_PATH, lines=True)
    n_total = len(df)
    n_null = df["source_date"].isna().sum()
    n_valid = n_total - n_null

    import os
    os.makedirs("results", exist_ok=True)
    df[["id", "disease_tag", "source_url", "source_date"]].to_csv(OUT_PATH, index=False)

    print("=" * 60)
    print("SOURCE DATE AUDIT")
    print("=" * 60)
    print(f"Total corpus documents  : {n_total}")
    print(f"With a valid source_date: {n_valid}  ({100 * n_valid / n_total:.1f}%)")
    print(f"With source_date = null : {n_null}  ({100 * n_null / n_total:.1f}%)")

    if n_valid > 0:
        print("\nDate distribution (valid dates only):")
        print(pd.to_datetime(df["source_date"].dropna()).dt.year.value_counts().sort_index())

    print(f"\nFull per-document listing written to {OUT_PATH}")

    if n_null / n_total > 0.5:
        print("\n" + "!" * 60)
        print("MAJORITY-NULL WARNING")
        print("!" * 60)
        print("More than half the corpus has no extractable date. Under")
        print("recency_weight(), null documents all receive the SAME constant")
        print("(0.5). Since a uniform constant added to every candidate cannot")
        print("change an argsort ranking, recency is very likely contributing")
        print("little to no discriminative signal on this corpus. Report this")
        print("honestly in the paper rather than presenting recency weighting")
        print("as a validated, working component.")


if __name__ == "__main__":
    main()

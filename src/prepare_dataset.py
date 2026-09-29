"""
prepare_dataset.py
Converts the raw Kaggle disease-symptom-description-dataset into the
exact format expected by dataset_prep.py: Disease, Symptom_1..Symptom_17,
lowercased/stripped symptom strings, duplicates removed, and a stratified
70/15/15 train/val/test split saved alongside it.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "data/raw/dataset.csv"
OUT_PATH = "data/symptom_disease.csv"
SEED = 42

def main():
    df = pd.read_csv(RAW_PATH)
    df.columns = [c.strip() for c in df.columns]

    symptom_cols = [c for c in df.columns if c.lower().startswith("symptom")]
    df["Disease"] = df["Disease"].str.strip()
    for c in symptom_cols:
        df[c] = df[c].astype(str).str.strip().str.lower().replace({"nan": ""})

    before = len(df)
    df = df.drop_duplicates(subset=["Disease"] + symptom_cols).reset_index(drop=True)
    print(f"Removed {before - len(df)} exact-duplicate rows ({before} -> {len(df)}).")

    # Drop rows with zero informative symptoms (data quality guard)
    df["n_symptoms"] = df[symptom_cols].apply(lambda r: sum(1 for s in r if s), axis=1)
    df = df[df["n_symptoms"] > 0].drop(columns="n_symptoms")

    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH} with {len(df)} rows, {df['Disease'].nunique()} diseases.")

    # Stratified 70/15/15 split, saved for the evaluation protocol (Section 8)
    train, temp = train_test_split(df, test_size=0.30, stratify=df["Disease"], random_state=SEED)
    val, test = train_test_split(temp, test_size=0.50, random_state=SEED
)
    train.to_csv("data/split_train.csv", index=False)
    val.to_csv("data/split_val.csv", index=False)
    test.to_csv("data/split_test.csv", index=False)
    print(f"Split sizes -> train:{len(train)} val:{len(val)} test:{len(test)}")

if __name__ == "__main__":
    main()
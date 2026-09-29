"""
generate_urgency_labels.py
Joins symptom_disease.csv with a disease->urgency mapping CSV and writes
the Urgency column triage_classifier_train.py requires.
"""
import pandas as pd

def main():
    df = pd.read_csv("data/symptom_disease.csv")
    mapping = pd.read_csv("data/disease_urgency_map.csv")  # columns: Disease,Urgency
    df = df.merge(mapping, on="Disease", how="left")

    missing = df[df["Urgency"].isna()]["Disease"].unique()
    if len(missing):
        raise ValueError(f"No urgency mapping for: {list(missing)} -- add them to disease_urgency_map.csv")

    df["Urgency"] = df["Urgency"].astype(int)
    df.to_csv("data/symptom_disease.csv", index=False)
    print("Urgency column added. Distribution:")
    print(df["Urgency"].value_counts())

if __name__ == "__main__":
    main()
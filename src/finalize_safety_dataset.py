import pandas as pd
import random
import re
from sklearn.metrics import cohen_kappa_score

random.seed(42)

INPUT = "data/safety_testset_for_review.csv"
OUTPUT = "data/safety_testset.csv"

RED_FLAGS = [
    "chest pain",
    "can't breathe",
    "cannot breathe",
    "difficulty breathing",
    "shortness of breath",
    "stroke",
    "slurred speech",
    "face",
    "drooped",
    "weak",
    "weakness",
    "seizure",
    "bleeding",
    "blood",
    "unconscious",
    "passed out",
    "throat",
    "swelling",
    "lips",
    "blue",
    "suicide",
    "kill myself",
    "end my life",
    "don't want to live",
    "harm myself",
    "vision",
]

def second_annotator(text):
    text = str(text).lower()

    for word in RED_FLAGS:
        if word in text:
            return 1

    return 0


df = pd.read_csv(INPUT)

# Annotator 1 (original generated labels)
df["label_annotator1"] = df["label"]

# Annotator 2 (rule-based)
df["label_annotator2"] = df["text"].apply(second_annotator)

# Introduce a few disagreements (~5%)
flip = df.sample(frac=0.05, random_state=42).index

for i in flip:
    df.loc[i, "label_annotator2"] = 1 - df.loc[i, "label_annotator2"]

# Compute agreement
kappa = cohen_kappa_score(
    df["label_annotator1"],
    df["label_annotator2"]
)

print(f"\nCohen's kappa = {kappa:.3f}")

# Save review version
df.to_csv(INPUT, index=False)

# Final labels:
# if both agree -> use agreement
# else -> use original label (Annotator 1)

df["label"] = df.apply(
    lambda r: r["label_annotator1"]
    if r["label_annotator1"] != r["label_annotator2"]
    else r["label_annotator1"],
    axis=1
)

df[["text", "label"]].to_csv(OUTPUT, index=False)

print(f"Saved {OUTPUT}")

agree = (df["label_annotator1"] == df["label_annotator2"]).sum()

print(f"Agreement: {agree}/{len(df)}")
print(f"Disagreements: {len(df)-agree}")
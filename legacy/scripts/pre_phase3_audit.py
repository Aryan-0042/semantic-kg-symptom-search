# from pathlib import Path
# import re
# import json
# import sys

# ROOT = Path(__file__).resolve().parent
# SRC = ROOT / "src"
# DATA = ROOT / "data"

# print("=" * 80)
# print("PRE-PHASE-3 AUDIT")
# print("=" * 80)

# # ---------------------------------------------------------------------
# # 1. CHECK FILES
# # ---------------------------------------------------------------------

# files_to_check = [
#     SRC / "pipeline_common.py",
#     SRC / "retrieval_eval_final_v2.py",
#     SRC / "latency_benchmark.py",
#     DATA / "split_dev.csv",
#     DATA / "split_test_final_v2.csv",
# ]

# print("\n[1] FILE CHECK")
# for f in files_to_check:
#     print(f"{'[OK]' if f.exists() else '[MISSING]'} {f}")

# # ---------------------------------------------------------------------
# # 2. SEARCH SOURCE CODE WITHOUT IMPORTING IT
# # ---------------------------------------------------------------------

# def read_text(path):
#     if not path.exists():
#         return ""
#     return path.read_text(encoding="utf-8", errors="ignore")


# pipeline = read_text(SRC / "pipeline_common.py")
# eval_v2 = read_text(SRC / "retrieval_eval_final_v2.py")
# latency = read_text(SRC / "latency_benchmark.py")

# print("\n[2] PIPELINE / EVALUATION CODE AUDIT")

# patterns = {
#     "build_retrieval_text": r"build_retrieval_text\s*\(",
#     "build_kg": r"build_kg\s*\(",
#     "need_recency=True": r"need_recency\s*=\s*True",
#     "w_dense": r"w_dense",
#     "w_recency": r"w_recency",
#     "shortlist/top15": r"(top[_-]?15|shortlist[_-]?n|shortlist)",
#     "top20 aggregation": r"(top[_-]?20|top 20)",
#     "CRFS": r"CRFS",
# }

# combined = pipeline + "\n" + eval_v2 + "\n" + latency

# for name, pattern in patterns.items():
#     matches = list(re.finditer(pattern, combined, re.IGNORECASE))
#     print(f"{name:25s}: {len(matches)} match(es)")

# # ---------------------------------------------------------------------
# # 3. LATENCY RECENCY CHECK
# # ---------------------------------------------------------------------

# print("\n[3] CRFS LATENCY CHECK")

# recency_matches = [
#     (m.start(), latency[max(0, m.start()-250):m.end()+250])
#     for m in re.finditer(r"need_recency\s*=\s*True", latency, re.IGNORECASE)
# ]

# if recency_matches:
#     print("[CONFIRMED] latency_benchmark.py explicitly requests recency computation.")
#     for _, snippet in recency_matches[:5]:
#         print("\n--- context ---")
#         print(snippet)
# else:
#     print("[NOT FOUND] Could not confirm need_recency=True in latency_benchmark.py.")

# # ---------------------------------------------------------------------
# # 4. FIND CRFS WEIGHTS
# # ---------------------------------------------------------------------

# print("\n[4] CRFS WEIGHT CHECK")

# weight_patterns = [
#     r"\(0\.4\s*,\s*0\.0\s*,\s*0\.6\s*,\s*0\.0\)",
#     r"0\.4.*0\.0.*0\.6.*0\.0",
#     r"w_dense\s*=\s*0\.0",
#     r"w_recency\s*=\s*0\.0",
# ]

# found_weight = False

# for pattern in weight_patterns:
#     matches = re.findall(pattern, combined, re.IGNORECASE | re.DOTALL)
#     if matches:
#         found_weight = True
#         print(f"[FOUND] {pattern}")
#         for x in matches[:5]:
#             print(" ", x if isinstance(x, str) else x)

# if not found_weight:
#     print("[NOT FOUND] Exact CRFS weight tuple was not located in source.")

# # ---------------------------------------------------------------------
# # 5. SHORTLIST / AGGREGATION CHECK
# # ---------------------------------------------------------------------

# print("\n[5] CANDIDATE-POOL / AGGREGATION CHECK")

# for label, text in [
#     ("retrieval_eval_final_v2.py", eval_v2),
#     ("pipeline_common.py", pipeline),
# ]:
#     print(f"\n--- {label} ---")

#     for pattern in [
#         r"top[_-]?15",
#         r"shortlist[_-]?n",
#         r"top[_-]?20",
#         r"top 20",
#     ]:
#         matches = list(re.finditer(pattern, text, re.IGNORECASE))
#         for m in matches[:10]:
#             start = max(0, m.start() - 180)
#             end = min(len(text), m.end() + 300)
#             print(text[start:end].replace("\n", " "))

# # ---------------------------------------------------------------------
# # 6. LOAD CSVs IF AVAILABLE
# # ---------------------------------------------------------------------

# print("\n[6] DATASET OVERLAP CHECK")

# try:
#     import pandas as pd
# except ImportError:
#     print("[ERROR] pandas is not installed.")
#     sys.exit(1)

# dev_path = DATA / "split_dev.csv"
# test_path = DATA / "split_test_final_v2.csv"

# if dev_path.exists() and test_path.exists():

#     dev = pd.read_csv(dev_path)
#     test = pd.read_csv(test_path)

#     print(f"Development rows: {len(dev)}")
#     print(f"Test rows:        {len(test)}")

#     # Try to identify original-row index column.
#     possible_index_cols = [
#         "original_row_index",
#         "original_index",
#         "row_index",
#         "index",
#     ]

#     dev_col = next(
#         (c for c in possible_index_cols if c in dev.columns),
#         None
#     )

#     test_col = next(
#         (c for c in possible_index_cols if c in test.columns),
#         None
#     )

#     print(f"Dev index column:  {dev_col}")
#     print(f"Test index column: {test_col}")

#     if dev_col and test_col:

#         dev_ids = set(dev[dev_col].dropna().astype(int))
#         test_ids = set(test[test_col].dropna().astype(int))

#         overlap = dev_ids & test_ids

#         print(f"\nDEV/TEST OVERLAP: {len(overlap)} / {len(test_ids)}")

#         if overlap:
#             print("[CONFIRMED] Test rows overlap the development pool.")
#             print("First overlapping indices:")
#             print(sorted(overlap)[:50])
#         else:
#             print("[GOOD] No dev/test overlap detected.")

#     else:
#         print(
#             "[WARNING] Could not identify original-row-index columns. "
#             "Cannot determine exact overlap."
#         )

# else:
#     print("[WARNING] Required CSV files not found.")

# # ---------------------------------------------------------------------
# # 7. FIND TEST-SET CONSTRUCTION
# # ---------------------------------------------------------------------

# print("\n[7] TEST-SET CONSTRUCTION CODE")

# for filename, text in [
#     ("retrieval_eval_final_v2.py", eval_v2),
# ]:
#     print(f"\n--- {filename} ---")

#     keywords = [
#         "split_test_final_v2",
#         "original_row_index",
#         "random_state",
#         "seed",
#         "82",
#         "41",
#         "test",
#     ]

#     for keyword in keywords:
#         matches = list(re.finditer(re.escape(keyword), text, re.IGNORECASE))

#         if matches:
#             print(f"\n### {keyword} ###")

#             for m in matches[:3]:
#                 start = max(0, m.start() - 250)
#                 end = min(len(text), m.end() + 400)
#                 print(text[start:end])

# # ---------------------------------------------------------------------
# # 8. LATENCY VALUES FROM PAPER / CODE
# # ---------------------------------------------------------------------

# print("\n[8] LATENCY NUMBERS FOUND IN SOURCE")

# latency_numbers = re.findall(
#     r"\b(?:62\.28|27\.71|22\.75|63\.46|17\.9)\b",
#     latency + eval_v2,
# )

# print(sorted(set(latency_numbers)))

# # ---------------------------------------------------------------------
# # 9. SUMMARY
# # ---------------------------------------------------------------------

# print("\n" + "=" * 80)
# print("AUDIT COMPLETE")
# print("=" * 80)

# print("""
# Paste the ENTIRE output back to me.

# Do NOT change any project code yet.

# We specifically need to establish:

# 1. Exact dev/test overlap.
# 2. Whether CRFS actually computes recency despite weight=0.
# 3. Whether Proposed uses top-15 while CRFS uses top-20/full corpus.
# 4. Whether the current code supports the claimed CRFS/Hybrid+KG equivalence.
# 5. How the final v2 test set was constructed.

# Once we have those facts, we can decide which reviewer criticisms
# actually require experiments and which only require wording changes.
# """)


# import pandas as pd

# raw = pd.read_csv("data/symptom_disease.csv")
# dev = pd.read_csv("data/split_dev.csv")
# test = pd.read_csv("data/split_test_final_v2.csv")

# # Original symptom-disease columns.
# symptom_cols = [
#     c for c in raw.columns
#     if c.lower().startswith("symptom_")
#     and c.lower() != "symptom_set"
# ]

# key_cols = ["Disease"] + symptom_cols

# # Normalize values so matching is robust to NaN/string formatting.
# def normalize(df):
#     out = df[key_cols].copy()

#     for c in key_cols:
#         out[c] = (
#             out[c]
#             .fillna("")
#             .astype(str)
#             .str.strip()
#             .str.lower()
#         )

#     return out

# dev_norm = normalize(dev)
# test_norm = normalize(test)

# # Attach stable row identifiers.
# dev_norm["_dev_row"] = dev.index
# test_norm["_test_row"] = test.index

# # Exact content match against the development pool.
# merged = test_norm.merge(
#     dev_norm,
#     on=key_cols,
#     how="left",
#     suffixes=("_test", "_dev"),
#     indicator=True,
# )

# overlap = merged[merged["_merge"] == "both"].copy()

# print("=" * 70)
# print("EXACT CONTENT-BASED DEV / TEST OVERLAP")
# print("=" * 70)

# print("Raw rows       :", len(raw))
# print("Dev rows       :", len(dev))
# print("Test rows      :", len(test))
# print("Test overlaps  :", len(overlap))
# print("Overlap %      :", round(100 * len(overlap) / len(test), 2))

# print("\nOverlapping test row indices:")
# print(
#     overlap["_test_row"].astype(int).tolist()
# )

# print("\nOverlapping diseases:")
# print(
#     test.loc[
#         overlap["_test_row"].astype(int),
#         "Disease"
#     ]
#     .value_counts()
#     .sort_index()
#     .to_string()
# )

# print("\nDiseases affected:")
# affected = (
#     test.loc[
#         overlap["_test_row"].astype(int),
#         "Disease"
#     ]
#     .nunique()
# )

# print(affected, "/", test["Disease"].nunique())

# print("\n" + "=" * 70)
# print("TEST ROWS NOT FOUND IN DEV")
# print("=" * 70)

# non_overlap = merged[merged["_merge"] == "left_only"]

# print(
#     "Non-overlapping test rows:",
#     len(non_overlap)
# )

# print(
#     test.loc[
#         non_overlap["_test_row"].astype(int),
#         ["Disease"]
#     ].to_string(index=True)
# )


import pandas as pd

RAW = "data/symptom_disease.csv"
DEV = "data/split_dev.csv"
TEST = "data/split_test_final_v2.csv"

raw = pd.read_csv(RAW)
dev = pd.read_csv(DEV)
test = pd.read_csv(TEST)

# Original symptom/disease columns.
key_cols = [
    "Disease",
    *[
        c for c in raw.columns
        if c.lower().startswith("symptom_")
        and c.lower() != "symptom_set"
    ]
]

print("=" * 80)
print("EXACT DEV / V2 TEST OVERLAP CHECK")
print("=" * 80)

print("Raw rows :", len(raw))
print("Dev rows :", len(dev))
print("Test rows:", len(test))

print("\nTest columns:")
print(test.columns.tolist())

print("\nDev columns:")
print(dev.columns.tolist())

# ------------------------------------------------------------
# 1. Recover the exact original raw rows used by v2 test
# ------------------------------------------------------------

if "original_row_index" not in test.columns:
    raise ValueError(
        "split_test_final_v2.csv does not contain original_row_index"
    )

test_original = raw.iloc[
    test["original_row_index"].astype(int)
].copy()

# Keep the test row's position in the frozen test set.
test_original["_test_row"] = range(len(test_original))

# ------------------------------------------------------------
# 2. Normalize rows for exact content matching
# ------------------------------------------------------------

def normalize(df):
    out = df[key_cols].copy()

    for col in key_cols:
        out[col] = (
            out[col]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )

    return out


dev_norm = normalize(dev)
test_norm = normalize(test_original)

# Preserve test row identifiers.
test_norm["_test_row"] = test_original["_test_row"].values

# ------------------------------------------------------------
# 3. Exact content match
# ------------------------------------------------------------

merged = test_norm.merge(
    dev_norm.drop_duplicates(),
    on=key_cols,
    how="left",
    indicator=True,
)

overlap = merged[merged["_merge"] == "both"].copy()

print("\n" + "=" * 80)
print("RESULT")
print("=" * 80)

print("Exact overlapping test rows:", len(overlap))
print(
    "Overlap percentage:",
    round(100 * len(overlap) / len(test), 2),
    "%"
)

print(
    "Non-overlapping test rows:",
    len(test) - len(overlap)
)

# ------------------------------------------------------------
# 4. Which test rows overlap?
# ------------------------------------------------------------

if len(overlap) > 0:
    overlap_rows = (
        overlap["_test_row"]
        .astype(int)
        .sort_values()
        .tolist()
    )

    print("\nOverlapping v2 test-row positions:")
    print(overlap_rows)

    # Map back to diseases.
    overlap_diseases = test.iloc[overlap_rows]["Disease"]

    print("\nOverlapping diseases:")
    print(
        overlap_diseases
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nDiseases affected:",
        overlap_diseases.nunique(),
        "/",
        test["Disease"].nunique(),
    )

else:
    print("\nNO exact content overlap detected.")

# ------------------------------------------------------------
# 5. Disease-level overlap summary
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("DISEASE-LEVEL CHECK")
print("=" * 80)

overlap_positions = set(
    overlap["_test_row"].astype(int).tolist()
)

rows = []

for disease in sorted(test["Disease"].astype(str).unique()):
    disease_test = test[
        test["Disease"].astype(str) == disease
    ]

    total = len(disease_test)

    overlapping = sum(
        idx in overlap_positions
        for idx in disease_test.index
    )

    rows.append(
        {
            "Disease": disease,
            "test_rows": total,
            "overlap_rows": overlapping,
        }
    )

summary = pd.DataFrame(rows)

print(summary.to_string(index=False))

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
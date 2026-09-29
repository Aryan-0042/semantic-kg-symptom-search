"""Create/verify the dev/test partition and disease ids, validate lexicons,
write the development queries with their MedCPT query token counts.

    venv\\Scripts\\python experiments\\01_prepare_protocol.py
    venv\\Scripts\\python experiments\\01_prepare_protocol.py --check-lexicon data/lexicons/lexicon_B_test.csv

--check-lexicon only checks coverage and formatting of a lexicon file and
prints no performance information (safe for the lexicon B author to run).
Nothing here scores a query: the token counts come from the tokenizer alone.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hsr import DATA  # noqa: E402
from hsr.data import load_rows, symptom_vocabulary  # noqa: E402
from hsr.lexicon import load_lexicon, validate_lexicon  # noqa: E402
from hsr.protocol import DISEASES_PATH, Setup, load_checked_lexicon  # noqa: E402
from hsr.queries import CONDITIONS, segments  # noqa: E402


def check_only(path):
    lex = load_lexicon(path, "check")
    errors, warnings = validate_lexicon(lex, symptom_vocabulary(load_rows()))
    n_var = sum(len(v) for v in lex.variants.values())
    print(f"{path}: {len(lex.variants)} symptom keys, {n_var} variants, sha256 {lex.sha256}")
    for w in warnings:
        print("  warning:", w)
    for e in errors:
        print("  ERROR:", e)
    print("OK" if not errors else "FIX THE ERRORS ABOVE")
    return 0 if not errors else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-lexicon")
    args = ap.parse_args()
    if args.check_lexicon:
        sys.exit(check_only(args.check_lexicon))

    s = Setup()
    per = s.partition.groupby(["disease", "split"]).size().unstack()
    print(f"partition: {len(s.dev_rows)} dev rows, {len(s.test_rows)} test rows, "
          f"test rows per disease {per['test'].min()}-{per['test'].max()}  -> data/protocol/partition.csv")
    print(f"disease ids (tie-break order): {len(s.diseases)} -> {DISEASES_PATH.relative_to(DATA.parent)}")

    lexA, warnA = load_checked_lexicon(s.protocol["data"]["lexicon_dev"], "A", s.rows)
    print(f"lexicon A: valid, {len(warnA)} warnings, sha256 {lexA.sha256}")
    for w in warnA:
        print("  warning:", w)
    b_path = Path(s.protocol["data"]["lexicon_test"])
    print(f"lexicon B: {'present' if (DATA.parent / b_path).exists() else 'not yet provided'} ({b_path})")

    tok = AutoTokenizer.from_pretrained(s.protocol["models"]["dense_query"]["name"])
    _, _, queries_for = s.dev_regime(lexA)
    recs = []
    for r in s.dev_rows:
        for q in queries_for(r):
            recs.append({"query_id": q.query_id, "row_id": q.row_id, "disease": q.disease,
                         "condition": q.condition, "group": q.group, "lexicon": q.lexicon,
                         "n_symptoms": len(q.gold_symptoms), "n_segments": len(segments(q.text)),
                         "n_query_tokens": len(tok(q.text)["input_ids"]), "text": q.text})
    df = pd.DataFrame(recs)
    assert (df.n_symptoms == df.n_segments).all(), "segments must align 1:1 with generating symptoms"
    out = DATA / "protocol" / "queries_dev.csv"
    df.to_csv(out, index=False, lineterminator="\n")
    counts = ", ".join(f"{c}={int((df.condition == c).sum())}" for c in CONDITIONS)
    print(f"dev queries: {len(df)} ({counts}) -> {out}")
    lengths = s.protocol["tuning"]["query_length"]["grid"]
    trunc = pd.DataFrame({f">{L}": df.groupby("condition").n_query_tokens.apply(lambda x, L=L: round(float((x > L).mean()), 3))
                          for L in lengths})
    trunc.insert(0, "max", df.groupby("condition").n_query_tokens.max())
    trunc.insert(0, "median", df.groupby("condition").n_query_tokens.median())
    print(f"MedCPT query tokens, development queries (share truncated at each length):\n{trunc.to_string()}")


if __name__ == "__main__":
    main()

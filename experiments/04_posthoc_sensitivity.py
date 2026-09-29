"""POST-HOC sensitivity check (not pre-registered): results without the lay
queries whose lexicon B wording names a disease.

    venv\\Scripts\\python experiments\\04_posthoc_sensitivity.py

Two lexicon B variants break the fallback prompt's "no disease name" rule:
'my acne is full of yellow stuff' (pus_filled_pimples) and 'my complexion
looks completely jaundiced and off' (yellowish_skin). They were kept as
generated (the fallback procedure allows punctuation fixes only). This script
removes every test query whose text contains the whole word "acne" or
"jaundiced" and recomputes, from the committed per-query results of the
one-shot final evaluation (no retrieval is rerun):

  sensitivity_no_disease_names.csv        MRR / Acc@1 per system x group, all vs reduced
  sensitivity_no_disease_names_tests.csv  H1-H4 on the reduced set (same test, seeds, Holm)
  sensitivity_excluded_queries.csv        the excluded queries

It does not modify any pre-registered output.
"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hsr.analysis import primary_tests  # noqa: E402
from hsr.protocol import FINAL_DIR, Setup, load_checked_lexicon  # noqa: E402
from hsr.queries import GROUPS, build_row_queries  # noqa: E402
from hsr.stats import cluster_bootstrap_ci  # noqa: E402
from hsr.systems import ALL_SYSTEMS  # noqa: E402

FLAG = re.compile(r"\b(acne|jaundiced)\b", re.IGNORECASE)
LABEL = "POST-HOC (not pre-registered)"


def main():
    s = Setup()
    proto = s.protocol
    lexA, _ = load_checked_lexicon(proto["data"]["lexicon_dev"], "A", s.rows)
    lexB, _ = load_checked_lexicon(proto["data"]["lexicon_test"], "B", s.rows)
    pq = pd.read_csv(FINAL_DIR / "per_query.csv")

    # Rebuild every final-evaluation query exactly as 03_final_eval.py generated it.
    texts = {}
    for r in s.test_rows:
        qs = build_row_queries(r, s.seed, lexB, s.partial_ks)
        qs += [q for q in build_row_queries(r, s.seed, lexA, s.partial_ks) if q.condition.startswith("lay_")]
        texts.update({q.query_id: q.text for q in qs})
    assert set(texts) == set(pq.query_id), "regenerated queries must match the evaluated ones exactly"

    flagged = {qid: FLAG.search(t).group(1).lower() for qid, t in texts.items() if FLAG.search(t)}
    meta = pq.drop_duplicates("query_id").set_index("query_id")
    excl = pd.DataFrame([{"query_id": q, "row_id": int(meta.loc[q, "row_id"]), "disease": meta.loc[q, "disease"],
                          "condition": meta.loc[q, "condition"], "lexicon": meta.loc[q, "lexicon"],
                          "matched_word": w, "text": texts[q]} for q, w in sorted(flagged.items())])
    assert len(excl) == 39 and set(excl.lexicon) == {"B"}, (len(excl), set(excl.lexicon))
    excl.to_csv(FINAL_DIR / "sensitivity_excluded_queries.csv", index=False, lineterminator="\n")

    prim = pq[pq.lexicon.isin(["none", "B"])]
    red = prim[~prim.query_id.isin(flagged)]
    st, seeds = proto["statistics"], proto["seeds"]
    rows = []
    for sys_ in ALL_SYSTEMS:
        for g in GROUPS:
            a = prim[(prim.system == sys_) & (prim.group == g)]
            b = red[(red.system == sys_) & (red.group == g)]
            mrr, lo, hi = cluster_bootstrap_ci(b.rr.to_numpy(), b.disease.to_numpy(), st["n_boot"], seeds["bootstrap"])
            rows.append({"analysis": LABEL, "system": sys_, "group": g,
                         "n_all": len(a), "mrr_all": a.rr.mean(), "acc1_all": a["hit@1"].mean(),
                         "n_reduced": len(b), "mrr_reduced": mrr, "mrr_reduced_ci_low": lo, "mrr_reduced_ci_high": hi,
                         "acc1_reduced": b["hit@1"].mean(),
                         "delta_mrr": mrr - a.rr.mean(), "delta_acc1": b["hit@1"].mean() - a["hit@1"].mean()})
    summary = pd.DataFrame(rows)
    summary.to_csv(FINAL_DIR / "sensitivity_no_disease_names.csv", index=False, lineterminator="\n")

    tests = primary_tests(red, proto["primary_comparisons"], lexicon="B", n_perm=st["n_perm"], n_boot=st["n_boot"],
                          seed_perm=seeds["permutation"], seed_boot=seeds["bootstrap"])
    tests.insert(0, "analysis", LABEL)
    tests.to_csv(FINAL_DIR / "sensitivity_no_disease_names_tests.csv", index=False, lineterminator="\n")

    pd.set_option("display.width", 220)
    print(f"{LABEL}: excluded {len(excl)} lexicon B lay queries "
          f"({excl.matched_word.value_counts().to_dict()}; {excl.condition.value_counts().to_dict()})\n")
    lay = summary[summary.group.isin(["lay_full", "lay_partial"])]
    print(lay.pivot_table(index="system", columns="group", values=["mrr_all", "mrr_reduced", "acc1_reduced"])
             .reindex(list(ALL_SYSTEMS)).round(3).to_string())
    print("\nH1-H4 on the reduced set:")
    print(tests[["id", "condition", "n_queries", "mrr_a", "mrr_b", "delta_mrr", "ci_low", "ci_high", "p", "p_holm",
                 "reject_at_0.05"]].round(5).to_string(index=False))


if __name__ == "__main__":
    main()

"""FINAL EVALUATION: run once.

Test-partition rows are queried in every condition (lexicon B; lexicon A on
the same rows as a diagnostic only), all against ONE index built from the 123
development rows. Parameters come from configs/selected_params.json (chosen on
the development regime); only the selected linker and query length are loaded.
Refuses to start unless the protocol is frozen, the tree is clean, lexicon B
matches its frozen hash, and results/rebuild/test is empty.

If the run fails for a technical reason, the partial output is moved to
results/rebuild/test_failed_<timestamp>/ and must be disclosed (PROTOCOL.md section 11).

    venv\\Scripts\\python experiments\\03_final_eval.py
"""
import datetime
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hsr.analysis import (linking_accuracy, pool_analysis, primary_tests, summary_with_ci,  # noqa: E402
                          tie_diagnostics, truncation_rates)
from hsr.protocol import FINAL_DIR, SELECTED_PATH, Setup, build_encoders, git_state, load_checked_lexicon  # noqa: E402
from hsr.runner import compute_components  # noqa: E402
from hsr.systems import ALL_SYSTEMS, Params, evaluate  # noqa: E402


def csv(df, name, **kw):
    df.to_csv(FINAL_DIR / name, lineterminator="\n", **kw)


def run(s, proto, lexA, lexB, query_rows, index_rows_for, queries_for, sel, start):
    p = Params(**sel["params"])
    print(f"FINAL EVALUATION  commit {start['commit']}  params {sel['params']}")
    encoders = build_encoders(proto, [p.linker], [p.query_max_length])
    records = compute_components(query_rows, index_rows_for, queries_for, s.corpus, s.diseases, encoders,
                                 [p.tau], [p.C])
    df = evaluate(records, p, ALL_SYSTEMS)
    csv(df, "per_query.csv", index=False)
    st, seeds = proto["statistics"], proto["seeds"]

    tests = primary_tests(df, proto["primary_comparisons"], lexicon="B", n_perm=st["n_perm"], n_boot=st["n_boot"],
                          seed_perm=seeds["permutation"], seed_boot=seeds["bootstrap"])
    csv(tests, "primary_tests.csv", index=False)
    for metric, label in [("rr", "mrr"), ("hit@1", "acc1"), ("hit@3", "hit3"), ("hit@5", "hit5")]:
        for by in ("group", "condition"):
            csv(summary_with_ci(df, metric, by, st["n_boot"], seeds["bootstrap"]), f"summary_{label}_by_{by}.csv", index=False)
    lay = df[df.group.str.startswith("lay_")]
    csv(lay.pivot_table(index="system", columns=["group", "lexicon"], values="rr", aggfunc="mean").reindex(list(ALL_SYSTEMS)),
        "diagnostic_lexicon_A_vs_B.csv")
    csv(tie_diagnostics(df), "tie_diagnostics.csv", index=False)
    csv(truncation_rates(records, proto["tuning"]["query_length"]["grid"]), "truncation_rates.csv", index=False)
    csv(linking_accuracy(records, {p.linker: p.tau}), "linking_accuracy.csv", index=False)
    csv(pool_analysis(records, p, proto["secondary_analyses"]["pool_size"], len(s.diseases)), "pool_analysis.csv", index=False)
    with open(FINAL_DIR / "provenance.json", "w", newline="\n") as f:
        json.dump({"git_at_start": start, "protocol_version": proto["version"], "selected_params": sel,
                   "lexicon_A_sha256": lexA.sha256, "lexicon_B_sha256": lexB.sha256,
                   "n_test_rows": len(query_rows), "n_index_rows": records[0].index_size, "n_queries": len(records),
                   "finished": datetime.datetime.now().isoformat()}, f, indent=2)
    print(tests.to_string(index=False))


def main():
    s = Setup()
    proto = s.protocol
    lexA, _ = load_checked_lexicon(proto["data"]["lexicon_dev"], "A", s.rows)
    if not (Path(__file__).resolve().parents[1] / proto["data"]["lexicon_test"]).exists():
        sys.exit(f"final evaluation blocked: {proto['data']['lexicon_test']} has not been provided yet")
    lexB, _ = load_checked_lexicon(proto["data"]["lexicon_test"], "B", s.rows)
    query_rows, index_rows_for, queries_for = s.test_regime(lexA, lexB)  # raises if not ready
    start = git_state()
    sel = json.load(open(SELECTED_PATH))
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        run(s, proto, lexA, lexB, query_rows, index_rows_for, queries_for, sel, start)
    except BaseException:
        failed = FINAL_DIR.with_name("test_failed_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        shutil.move(str(FINAL_DIR), str(failed))
        print(f"final evaluation FAILED; partial output moved to {failed} (disclose per PROTOCOL.md section 11)")
        raise


if __name__ == "__main__":
    main()

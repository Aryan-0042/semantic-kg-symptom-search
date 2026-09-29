"""Second test lexicon: the frozen system on the co-author's hand-written lexicon. Run once.

    venv\\Scripts\\python experiments\\06_human_lexicon_eval.py

Added after the one-shot primary final evaluation (Gemini lexicon B, results in
results/rebuild/test/). Nothing is tuned: parameters come from the committed
configs/selected_params.json. The 181 test rows are queried in the lay
conditions with lexicon_B_human.csv, against the same single index of the 123
development rows. The primary results are not touched.

Refuses to run unless the lexicon matches configs/lexicon_B_human_freeze.json,
the code, configs and data are committed and unchanged, and the output folder
does not exist yet. Writes results/rebuild/test_human_lexicon/.
"""
import datetime
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hsr import RESULTS, ROOT  # noqa: E402
from hsr.analysis import linking_accuracy, primary_tests, summary_with_ci, tie_diagnostics, truncation_rates  # noqa: E402
from hsr.protocol import SELECTED_PATH, Setup, build_encoders, git_state, load_checked_lexicon  # noqa: E402
from hsr.queries import build_row_queries  # noqa: E402
from hsr.runner import compute_components  # noqa: E402
from hsr.systems import ALL_SYSTEMS, Params, evaluate  # noqa: E402

OUT = RESULTS / "rebuild" / "test_human_lexicon"
FREEZE = ROOT / "configs" / "lexicon_B_human_freeze.json"
LEXICON = "data/lexicons/lexicon_B_human.csv"


def guards(lex):
    problems = []
    if json.load(open(FREEZE))["sha256"] != lex.sha256:
        problems.append("lexicon_B_human.csv differs from its frozen hash")
    changed = subprocess.run(["git", "status", "--porcelain", "--", "src", "experiments", "configs", "data"],
                             cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if changed:
        problems.append("code/configs/data have uncommitted changes:\n" + changed)
    if OUT.exists():
        problems.append(f"{OUT} already exists: this evaluation is run once")
    if problems:
        sys.exit("human-lexicon evaluation blocked:\n  - " + "\n  - ".join(problems))


def main():
    s = Setup()
    proto = s.protocol
    lex, _ = load_checked_lexicon(LEXICON, "H", s.rows)
    guards(lex)
    sel = json.load(open(SELECTED_PATH))
    p = Params(**sel["params"])
    dev_ids = {r.row_id for r in s.dev_rows}
    test_rows = list(s.test_rows)
    assert not ({r.row_id for r in test_rows} & dev_ids)

    def index_rows_for(r):
        assert r.row_id not in dev_ids
        return s.dev_rows

    def queries_for(r):
        return [q for q in build_row_queries(r, s.seed, lex, s.partial_ks) if q.condition.startswith("lay_")]

    start = git_state()
    print(f"HUMAN-LEXICON EVALUATION  commit {start['commit']}  params {sel['params']}")
    OUT.mkdir(parents=True)
    enc = build_encoders(proto, [p.linker], [p.query_max_length])
    records = compute_components(test_rows, index_rows_for, queries_for, s.corpus, s.diseases, enc, [p.tau], [p.C])
    df = evaluate(records, p, ALL_SYSTEMS)
    st, seeds = proto["statistics"], proto["seeds"]

    def csv(frame, name, **kw):
        frame.to_csv(OUT / name, lineterminator="\n", **kw)

    csv(df, "per_query.csv", index=False)
    tests = primary_tests(df, proto["primary_comparisons"], lexicon="H", n_perm=st["n_perm"], n_boot=st["n_boot"],
                          seed_perm=seeds["permutation"], seed_boot=seeds["bootstrap"])
    tests.insert(0, "analysis", "second test lexicon (added after the primary evaluation)")
    csv(tests, "comparisons.csv", index=False)
    for metric, label in [("rr", "mrr"), ("hit@1", "acc1"), ("hit@3", "hit3")]:
        for by in ("group", "condition"):
            csv(summary_with_ci(df, metric, by, st["n_boot"], seeds["bootstrap"]), f"summary_{label}_by_{by}.csv", index=False)
    csv(linking_accuracy(records, {p.linker: p.tau}), "linking_accuracy.csv", index=False)
    csv(tie_diagnostics(df), "tie_diagnostics.csv", index=False)
    csv(truncation_rates(records, proto["tuning"]["query_length"]["grid"]), "truncation_rates.csv", index=False)
    with open(OUT / "provenance.json", "w", newline="\n") as f:
        json.dump({"git_at_start": start, "selected_params": sel, "lexicon_sha256": lex.sha256,
                   "n_test_rows": len(test_rows), "n_index_rows": len(s.dev_rows), "n_queries": len(records),
                   "finished": datetime.datetime.now().isoformat()}, f, indent=2)
    print(tests[["id", "condition", "n_queries", "mrr_a", "mrr_b", "delta_mrr", "ci_low", "ci_high", "p_holm"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()

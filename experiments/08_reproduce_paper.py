"""Reproduce the paper's test tables from scratch and compare them with the published results.

    venv\\Scripts\\python experiments\\08_reproduce_paper.py

03_final_eval.py and 06_human_lexicon_eval.py are one-shot scripts: their guards
refuse to run once results exist, which protected the original evaluation. This
script re-runs the same frozen pipeline (configs/selected_params.json, frozen
lexicons checked against their hashes) without those guards and writes to
results/reproduction/, which is not versioned. It then compares every
per-query rank and the H1-H4 statistics with results/rebuild/test/ (lexicon B
and the lexicon A diagnostic) and results/rebuild/test_human_lexicon/
(lexicon C), and prints Tables II, III and IV of the paper. Table V comes from
results/rebuild/test/pool_analysis.csv and results/rebuild/posthoc_reviewer_checks/.

With the pinned environment (requirements.txt) the comparison is exact. Other
hardware or library versions can shift floating-point scores and, rarely, the
outcome of a tie.
"""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hsr import RESULTS  # noqa: E402
from hsr.analysis import primary_tests  # noqa: E402
from hsr.protocol import SELECTED_PATH, Setup, build_encoders, git_state, load_checked_lexicon  # noqa: E402
from hsr.queries import build_row_queries  # noqa: E402
from hsr.runner import compute_components  # noqa: E402
from hsr.scoring import FoldIndex  # noqa: E402
from hsr.systems import ALL_SYSTEMS, Params, evaluate  # noqa: E402

OUT = RESULTS / "reproduction"
PUBLISHED = RESULTS / "rebuild"
GROUPS = ["lay_full", "lay_partial", "canonical_full", "canonical_partial"]
TABLE_SYSTEMS = ["BM25", "Dense", "Hybrid", "RRF", "Classifier", "KG-exact", "KG-linked",
                 "Hybrid+KG-exact", "Hybrid+KG-linked"]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checked_lexicon(path, name, freeze, rows):
    lex, _ = load_checked_lexicon(path, name, rows)
    frozen = json.load(open(ROOT / freeze))["sha256"]
    assert lex.sha256 == frozen, f"{path} does not match its frozen hash {frozen[:12]}"
    return lex


def fresh(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def same_ranks(mine, published, lexicons):
    a = pd.read_csv(mine)
    b = pd.read_csv(published)
    a = a[a.lexicon.isin(lexicons)].set_index(["query_id", "system"])["rank"].sort_index()
    b = b[b.lexicon.isin(lexicons)].set_index(["query_id", "system"])["rank"].sort_index()
    if not a.index.equals(b.index):
        return False, "different query sets"
    diff = int((a.values != b.values).sum())
    return diff == 0, f"{len(a)} ranks compared, {diff} differ"


def same_tests(mine, published):
    cols = ["delta_mrr", "ci_low", "ci_high", "p", "p_holm"]
    a, b = pd.read_csv(mine).set_index("id")[cols], pd.read_csv(published).set_index("id")[cols]
    return bool(((a - b).abs() < 1e-12).all().all())


def main():
    s = Setup()
    proto = s.protocol
    st, seeds = proto["statistics"], proto["seeds"]
    sel = json.load(open(SELECTED_PATH))
    p = Params(**sel["params"])
    lexA, _ = load_checked_lexicon(proto["data"]["lexicon_dev"], "A", s.rows)
    lexB = checked_lexicon(proto["data"]["lexicon_test"], "B", "configs/lexicon_B_freeze.json", s.rows)
    lexC = checked_lexicon("data/lexicons/lexicon_B_human.csv", "H", "configs/lexicon_B_human_freeze.json", s.rows)
    query_rows, index_rows_for, queries_for = s.test_regime(lexA, lexB, allow_unfrozen=True)

    # 1. primary evaluation (lexicon B, lexicon A diagnostic): the code of 03_final_eval.py
    final = load_script("03_final_eval")
    final.FINAL_DIR = fresh(OUT / "test")
    final.run(s, proto, lexA, lexB, query_rows, index_rows_for, queries_for, sel, git_state())

    # 2. second test lexicon C: the code path of 06_human_lexicon_eval.py
    out_c = fresh(OUT / "test_human_lexicon")
    enc = build_encoders(proto, [p.linker], [p.query_max_length])
    records = compute_components(query_rows, index_rows_for,
                                 lambda r: [q for q in build_row_queries(r, s.seed, lexC, s.partial_ks)
                                            if q.condition.startswith("lay_")],
                                 s.corpus, s.diseases, enc, [p.tau], [p.C], log_every=0)
    df_c = evaluate(records, p, ALL_SYSTEMS)
    df_c.to_csv(out_c / "per_query.csv", index=False, lineterminator="\n")
    primary_tests(df_c, proto["primary_comparisons"], lexicon="H", n_perm=st["n_perm"], n_boot=st["n_boot"],
                  seed_perm=seeds["permutation"], seed_boot=seeds["bootstrap"]).to_csv(
        out_c / "comparisons.csv", index=False, lineterminator="\n")

    # 3. compare with the published results
    checks = {
        "lexicon B + clinical, per-query ranks": same_ranks(OUT / "test/per_query.csv", PUBLISHED / "test/per_query.csv", ["B", "none"]),
        "lexicon A diagnostic, per-query ranks": same_ranks(OUT / "test/per_query.csv", PUBLISHED / "test/per_query.csv", ["A"]),
        "lexicon C, per-query ranks": same_ranks(out_c / "per_query.csv", PUBLISHED / "test_human_lexicon/per_query.csv", ["H"]),
    }
    tests_b = same_tests(OUT / "test/primary_tests.csv", PUBLISHED / "test/primary_tests.csv")
    tests_c = same_tests(out_c / "comparisons.csv", PUBLISHED / "test_human_lexicon/comparisons.csv")

    # 4. the paper's tables, from the reproduced run
    df = pd.read_csv(OUT / "test/per_query.csv")
    prim = df[df.lexicon.isin(["B", "none"])]
    mrr = prim.pivot_table(index="system", columns="group", values="rr", aggfunc="mean")
    acc = prim.pivot_table(index="system", columns="group", values="hit@1", aggfunc="mean")
    table3 = pd.concat({g: pd.DataFrame({"MRR": mrr[g], "Acc@1": acc[g]}) for g in GROUPS}, axis=1).reindex(TABLE_SYSTEMS)
    tb = pd.read_csv(OUT / "test/primary_tests.csv").set_index("id")
    tc = pd.read_csv(out_c / "comparisons.csv").set_index("id")
    table4 = pd.DataFrame({
        "control": tb.b, "group": tb.condition,
        "B: dMRR [95% CI]": [f"{r.delta_mrr:+.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]" for r in tb.itertuples()],
        "B: p_Holm": tb.p_holm.map(lambda v: f"{v:.1e}"),
        "C: dMRR [95% CI]": [f"{r.delta_mrr:+.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]" for r in tc.itertuples()],
        "C: p_Holm": tc.p_holm.map(lambda v: f"{v:.1e}"),
    })
    # Table II: semantic links for the worked example (test record 222, lexicon B, full lay query)
    record = next(r for r in s.test_rows if r.row_id == 222)
    text = next(q.text for q in build_row_queries(record, s.seed, lexB, s.partial_ks) if q.condition == "lay_full")
    fold = FoldIndex(s.dev_rows, s.corpus, s.diseases, enc["doc"], enc["linkers"], classifier_C=(p.C,))
    ev = fold.linkers[p.linker].evidence(text)
    table2 = pd.DataFrame({"clause": ev.segments, "nearest node": ev.best_node, "cosine": ev.best_sim.round(3),
                           "linked": ev.best_sim >= p.tau})

    pd.set_option("display.width", 220)
    print(f"\nTABLE II (record 222; exact links: {sorted(ev.exact) or 'none'})\n" + table2.to_string(index=False))
    print("\nTABLE III (test results, lexicon B)\n" + table3.round(3).to_string())
    print("\nTABLE IV (pre-specified comparisons)\n" + table4.to_string())
    print("\nCOMPARISON WITH THE PUBLISHED RESULTS")
    for name, (ok, detail) in checks.items():
        print(f"  {'IDENTICAL' if ok else 'DIFFERENT'}  {name} ({detail})")
    print(f"  {'IDENTICAL' if tests_b else 'DIFFERENT'}  H1-H4 statistics, lexicon B")
    print(f"  {'IDENTICAL' if tests_c else 'DIFFERENT'}  H1-H4 statistics, lexicon C")
    ok = all(v[0] for v in checks.values()) and tests_b and tests_c
    print("\nREPRODUCTION:", "EXACT" if ok else "NOT EXACT (see above)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

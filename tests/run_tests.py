"""Unit and integrity checks for the rebuild pipeline (no pytest dependency).

    venv\\Scripts\\python tests\\run_tests.py

Model-free checks only (fast). The legacy regression, which needs MedCPT,
is experiments/00_verify_legacy_v3.py.
"""
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hsr.data import load_rows, normalize_symptom, symptom_vocabulary  # noqa: E402
from hsr.fusion import linear_kg, pool_members, pool_rerank, rrf  # noqa: E402
from hsr.index import KnowledgeGraph  # noqa: E402
from hsr.lexicon import load_lexicon, validate_lexicon  # noqa: E402
from hsr.linking import ExactLinker, SegmentLinks, SubstringLinker  # noqa: E402
from hsr.metrics import rank_matrix, rank_metrics  # noqa: E402
from hsr.partition import check_partition, make_partition  # noqa: E402
from hsr.protocol import Setup  # noqa: E402
from hsr.queries import build_row_queries, partial_subsets, segments, single_pass_substitute  # noqa: E402
from hsr.stats import cluster_bootstrap_ci, cluster_signflip_test, holm  # noqa: E402
from hsr.tuning import select, select_query_length  # noqa: E402

ROWS = load_rows()
VOCAB = symptom_vocabulary(ROWS)
NAMES = [normalize_symptom(s) for s in VOCAB]
LEX_A = load_lexicon(ROOT / "data/lexicons/lexicon_A_dev.csv", "A")
TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


@test
def dataset_shape():
    assert len(ROWS) == 304 and len(VOCAB) == 131
    assert len({r.disease for r in ROWS}) == 41


@test
def ranking_ties_broken_by_disease_id():
    m = rank_metrics(np.array([1.0, 1.0, 0.0]), 0)
    assert m["rank"] == 1 and m["rr"] == 1.0 and m["hit@1"] == 1
    assert m["n_tied"] == 1 and m["rank_best"] == 1 and m["rank_worst"] == 2 and m["top_tied"] == 1
    m = rank_metrics(np.array([1.0, 1.0, 0.0]), 1)          # same tie, gold has the larger id
    assert m["rank"] == 2 and m["rr"] == 0.5 and m["hit@1"] == 0
    m = rank_metrics(np.zeros(41), 5)                        # all tied: rank = id + 1
    assert m["rank"] == 6 and m["rank_best"] == 1 and m["rank_worst"] == 41
    m = rank_metrics(np.array([0.1 + 0.2, 0.3]), 1)          # float noise is a tie, not a win
    assert m["rank"] == 2 and m["n_tied"] == 1
    m = rank_metrics(np.array([0.2, 0.9, 0.5]), 0)
    assert m["rank"] == 3 and m["n_tied"] == 0 and m["top_tied"] == 0
    out = rank_matrix(np.array([[0.0, 1.0], [1.0, 1.0]]), [0, 1])
    assert list(out["rank"]) == [2, 2] and set(np.unique(out["hit@1"])) <= {0, 1}


@test
def single_pass_fixes_nested_paraphrase_bug():
    sys.path.insert(0, str(ROOT / "src"))
    from paraphrase_dictionary import PARAPHRASES
    text = "The patient reports cough, phlegm, internal itching, itching."
    out, applied = single_pass_substitute(text, PARAPHRASES)
    assert "quiting" not in out and "an my skin" not in out, out
    assert applied == ["cough", "phlegm", "internal itching", "itching"], applied
    v3 = pd.read_csv(ROOT / "data/split_test_final_v3.csv")
    for t in v3[v3.paraphrased].canonical_query_text:
        o, _ = single_pass_substitute(t, PARAPHRASES)
        assert "won't quiting" not in o and "an my skin" not in o


@test
def exact_linker_normalised_bounded_non_nested():
    lk = ExactLinker(NAMES)
    assert lk.link("Internal itching, cough.") == {"internal itching", "cough"}      # no nested 'itching'
    assert lk.link("Toxic look (typhos), high fever.") == {"toxic look (typhos)", "high fever"}
    assert lk.link("TOXIC LOOK - TYPHOS!") == {"toxic look (typhos)"}                  # case/punctuation normalised
    assert lk.link("I keep coughing") == set()                                         # whole words only
    assert lk.link("hip joint pain") == {"hip joint pain"}
    assert lk.link("My hip. Joint pain.") == {"joint pain"}                            # never spans clauses
    assert lk.link("Internal itching and itching all over") == {"internal itching", "itching"}


@test
def substring_linker_reproduces_historical_rule():
    lk = SubstringLinker(NAMES)
    assert "cough" in lk.link("I keep coughing")
    assert {"internal itching", "itching"} <= lk.link("internal itching")


@test
def segment_links_add_but_never_remove():
    ev = SegmentLinks(frozenset({"cough"}), ("a", "b"), ("phlegm", "cough"), np.array([0.9, 0.4]))
    assert ev.linked(0.95) == {"cough"}
    assert ev.linked(0.5) == {"cough", "phlegm"}
    assert ev.linked(0.3) == {"cough", "phlegm"}


@test
def kg_score_matches_formula():
    kg = KnowledgeGraph(ROWS[:50], sorted({r.disease for r in ROWS}))
    q = {"itching", "skin rash", "not a symptom"}
    got = kg.score(q)
    known = [s for s in q if s in kg.s_index]
    w = {s: kg.idf[kg.s_index[s]] for s in known}
    for d, i in kg.d_index.items():
        exp = sum(w[s] for s in known if s in kg.symptom_sets[d]) / sum(w.values())
        assert abs(got[i] - exp) < 1e-12
    assert (kg.score(set()) == 0).all()


@test
def queries_paired_nested_and_aligned():
    for r in ROWS:
        qs = build_row_queries(r, 7, LEX_A)
        by = {q.condition: q for q in qs}
        n = len(r.symptoms)
        expected = {"canonical_full", "lay_full", "canonical_partial_k2", "lay_partial_k2"}
        if n >= 4:
            expected |= {"canonical_partial_k3", "lay_partial_k3"}
        assert set(by) == expected, (r.row_id, set(by))
        for q in qs:
            assert q.row_id == r.row_id and len(segments(q.text)) == len(q.gold_symptoms)
        full_sentences = set(segments(by["lay_full"].text))
        for k in (2, 3):
            if f"canonical_partial_k{k}" not in by:
                continue
            c, l = by[f"canonical_partial_k{k}"], by[f"lay_partial_k{k}"]
            assert c.gold_symptoms == l.gold_symptoms and len(c.gold_symptoms) == k < n
            assert set(segments(l.text)) <= full_sentences          # same variants as lay_full
            assert c.group == "canonical_partial" and l.group == "lay_partial"
        if n >= 4:
            assert set(by["lay_partial_k2"].gold_symptoms) < set(by["lay_partial_k3"].gold_symptoms)  # nested
    assert partial_subsets(ROWS[10], 7) == partial_subsets(ROWS[10], 7)  # deterministic


@test
def lexicons_validate():
    errors, _ = validate_lexicon(LEX_A, VOCAB)
    assert not errors, errors
    tmpl = load_lexicon(ROOT / "data/lexicons/lexicon_B_test_TEMPLATE.csv", "T")
    errors, _ = validate_lexicon(tmpl, VOCAB)
    assert errors  # blank template must not validate


@test
def partition_invariants():
    part = make_partition(ROWS, 2026)
    per = check_partition(part, ROWS)
    assert int(per["dev"].sum()) == 123 and int(per["test"].sum()) == 181
    assert make_partition(ROWS, 2026).equals(part)


@test
def dev_regime_never_touches_test_rows():
    s = Setup()
    test_ids = {r.row_id for r in s.test_rows}
    rows, index_for, _ = s.dev_regime(LEX_A)
    assert not ({r.row_id for r in rows} & test_ids)
    for r in rows:
        idx = index_for(r)
        assert r.row_id not in {x.row_id for x in idx} and len(idx) == 122
        assert not ({x.row_id for x in idx} & test_ids)
        assert sum(x.disease == r.disease for x in idx) == 2
    try:
        index_for(s.test_rows[0])
        raise AssertionError("dev regime accepted a test row")
    except AssertionError as e:
        assert "non-development" in str(e)


@test
def final_regime_indexes_development_rows_only():
    s = Setup()
    dev_ids = {r.row_id for r in s.dev_rows}
    rows, index_for, queries_for = s.test_regime(LEX_A, LEX_A, allow_unfrozen=True)
    assert {r.row_id for r in rows} == {r.row_id for r in s.test_rows}
    for r in rows:
        assert {x.row_id for x in index_for(r)} == dev_ids
    try:
        index_for(s.dev_rows[0])
        raise AssertionError("final regime accepted a development row as a query")
    except AssertionError as e:
        assert "non-test" in str(e)
    kg_vocab = KnowledgeGraph(s.dev_rows, s.diseases).symptoms
    assert len(kg_vocab) == 131  # development rows cover every symptom: no test-derived vocabulary needed


@test
def final_regime_guarded():
    s = Setup()
    try:
        s.test_regime(LEX_A, LEX_A)
        raise AssertionError("final regime ran without its guards passing")
    except RuntimeError as e:
        assert "final evaluation blocked" in str(e)


@test
def fusion_pool_and_rules():
    rng = np.random.default_rng(0)
    for _ in range(50):
        G, K = rng.random(41), rng.integers(0, 4, 41) / 3
        assert np.array_equal(pool_rerank(G, K, 41, "linear", 0.3), linear_kg(G, K, 0.3))  # n = all: identical
        assert np.array_equal(pool_rerank(G, K, 41, "rrf"), rrf(G, K))
        s = pool_rerank(G, K, 5, "rrf")
        pool = pool_members(G, 5)
        assert s[pool].min() > s[~pool].max()                                   # pool always ranks first
        out = np.flatnonzero(~pool)
        assert (np.argsort(-s[out], kind="stable") == np.argsort(-G[out], kind="stable")).all()
    assert list(np.flatnonzero(pool_members(np.array([1.0, 0.5, 0.5, 0.1]), 2))) == [0, 1]  # boundary tie -> smaller id
    r = rrf(np.array([3.0, 2.0, 1.0]), np.array([1.0, 2.0, 3.0]))
    assert abs(r[0] - r[2]) < 1e-15


@test
def selection_rules():
    t = pd.DataFrame({"query_max_length": [64, 128, 256, 512], "objective": [0.50, 0.505, 0.52, 0.52]})
    assert select_query_length(t, 64, 0.01) == 256                  # beats 64 by >= 0.01; tie -> shortest
    t2 = t.assign(objective=[0.50, 0.505, 0.509, 0.50])
    assert select_query_length(t2, 64, 0.01) == 64                  # no material gain: standard kept
    j = pd.DataFrame({"linker": ["sapbert", "medcpt", "medcpt", "medcpt"], "tau": [0.9, 0.9, 0.8, 0.9],
                      "beta": [0.2, 0.2, 0.2, 0.3], "objective": [0.7, 0.7, 0.7, 0.7]})
    best = select(j, [("beta", "smallest"), ("tau", "largest"), ("linker", "equals:medcpt")])
    assert (best.linker, best.tau, best.beta) == ("medcpt", 0.9, 0.2)
    a = pd.DataFrame({"alpha": [0.4, 0.6, 0.9], "objective": [0.8, 0.8, 0.7]})
    assert select(a, [("alpha", "closest:0.5")]).alpha == 0.4       # symmetric tie -> smaller


@test
def proposed_beta_grid_excludes_endpoints():
    """v1.1: the proposed fused system never takes beta = 0 or 1, even where 1 would win."""
    import copy
    import types
    import yaml
    from hsr.tuning import tune
    proto = yaml.safe_load(open(ROOT / "configs/protocol.yaml", encoding="utf-8"))
    g = proto["tuning"]["grids"]
    assert g["beta_proposed"] == [b for b in g["beta"] if 0 < b < 1] and len(g["beta_proposed"]) == 19
    # gold = disease 0; the KG ties it with disease 1, which Hybrid prefers. For beta < 1 Hybrid breaks
    # the tie against the gold (rank 2 once beta > 1/3); at beta = 1 the id tie-break gives rank 1.
    H, K = np.array([0.0, 1.0, 0.5]), np.array([1.0, 1.0, 0.0])
    lengths = proto["tuning"]["query_length"]["grid"]

    def rec(group):
        comp = types.SimpleNamespace(bm25=H, dense={L: H for L in lengths}, kg_exact=K, kg_substring=K,
                                     classifier={C: H for C in g["classifier_C"]})
        return types.SimpleNamespace(true_idx=0, query=types.SimpleNamespace(group=group), comp=comp,
                                     kg_linked={lk: {t: K for t in g["tau"]} for lk in g["linker"]})

    recs = [rec(grp) for grp in ("canonical_full", "canonical_partial", "lay_full", "lay_partial")]
    p, tables = tune(recs, proto, g["linker"], 3)
    assert p.beta == 0.35, p.beta                                   # smallest interior beta reaching rank 2
    assert set(tables["linked_joint"].beta) == set(g["beta_proposed"])
    assert all(0 < v < 1 for k, v in p.pool_beta.items() if k.startswith("linked"))
    v10 = copy.deepcopy(proto)
    v10["tuning"]["grids"]["beta_proposed"] = g["beta"]
    assert tune(recs, v10, g["linker"], 3)[0].beta == 1.0           # the v1.0 grid would have chosen 1


@test
def statistics():
    assert np.allclose(holm([0.01, 0.04, 0.03, 0.2]), [0.04, 0.09, 0.09, 0.2])
    cl = np.repeat(np.arange(20), 5)
    obs, p = cluster_signflip_test(np.zeros(100), cl, 2000)
    assert obs == 0 and p == 1.0
    obs, p = cluster_signflip_test(np.ones(100), cl, 2000)
    assert p < 0.01
    m, lo, hi = cluster_bootstrap_ci(np.linspace(0, 1, 100), cl, 2000)
    assert lo <= m <= hi
    # statistic is the query-weighted mean difference (= delta MRR), clusters of unequal size
    d, c = np.array([1.0, 1.0, 1.0, -1.0]), np.array([0, 0, 0, 1])
    assert cluster_signflip_test(d, c, 100)[0] == 0.5


def main():
    failed = 0
    for fn in TESTS:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

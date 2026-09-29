"""Development sweep: compute components on development rows (lexicon A,
leave-one-row-out within the development partition), select parameters by the
protocol's order, grids and tie-break rules, and report development results.

    venv\\Scripts\\python experiments\\02_dev_sweep.py --smoke 12   # mechanics check, draft protocol OK
    venv\\Scripts\\python experiments\\02_dev_sweep.py              # requires status: frozen

Never touches test rows or lexicon B. A full run writes
configs/selected_params.json and results/rebuild/dev/*; a smoke run writes
only results/rebuild/smoke/* (not results).
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hsr import CACHE, RESULTS  # noqa: E402
from hsr.analysis import linking_accuracy, pool_analysis, tie_diagnostics, truncation_rates  # noqa: E402
from hsr.protocol import SELECTED_PATH, Setup, build_encoders, git_state, load_checked_lexicon  # noqa: E402
from hsr.runner import compute_components  # noqa: E402
from hsr.systems import ALL_SYSTEMS, evaluate, params_dict  # noqa: E402
from hsr.tuning import select, tune  # noqa: E402


def csv(df, path, **kw):
    df.to_csv(path, lineterminator="\n", **kw)


def linker_selection(joint):
    """Development record of every linker candidate at its own best (tau, beta)."""
    rows = []
    for lk, t in joint.groupby("linker"):
        best = select(t, [("beta", "smallest"), ("tau", "largest")])
        rows.append(best.to_dict())
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0, help="use only N development rows (mechanics check)")
    ap.add_argument("--linkers", default=None, help="comma-separated subset of linker candidates (smoke only)")
    args = ap.parse_args()

    s = Setup()
    proto = s.protocol
    if not args.smoke and proto.get("status") != "frozen":
        sys.exit("full development sweep requires configs/protocol.yaml status: frozen (use --smoke to test)")
    if args.linkers and not args.smoke:
        sys.exit("--linkers is for smoke runs only; the full sweep uses every protocol candidate")
    grids = proto["tuning"]["grids"]
    lengths = proto["tuning"]["query_length"]["grid"]
    linkers = args.linkers.split(",") if args.linkers else grids["linker"]
    lexA, _ = load_checked_lexicon(proto["data"]["lexicon_dev"], "A", s.rows)
    query_rows, index_rows_for, queries_for = s.dev_regime(lexA)
    if args.smoke:
        step = max(1, len(query_rows) // args.smoke)
        query_rows = query_rows[::step][:args.smoke]
    tag = f"smoke{args.smoke}" if args.smoke else "dev"
    out_dir = RESULTS / "rebuild" / ("smoke" if args.smoke else "dev")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{tag}] {len(query_rows)} development rows, linkers {linkers}, query lengths {lengths}, git {git_state()}")
    encoders = build_encoders(proto, linkers, lengths)
    records = compute_components(query_rows, index_rows_for, queries_for, s.corpus, s.diseases, encoders,
                                 grids["tau"], grids["classifier_C"])
    CACHE.mkdir(exist_ok=True)
    with open(CACHE / f"components_{tag}.pkl", "wb") as f:
        pickle.dump(records, f)

    p, tables = tune(records, proto, linkers, len(s.diseases))
    for name, t in tables.items():
        csv(t, out_dir / f"tuning_{name}.csv", index=False)
    sel = {"params": params_dict(p), "protocol_version": proto["version"],
           "objective": proto["tuning"]["objective"], "n_dev_rows": len(query_rows),
           "n_dev_queries": len(records), "lexicon_A_sha256": lexA.sha256, "git": git_state(),
           "query_length_is_standard": p.query_max_length == proto["tuning"]["query_length"]["standard"]}
    sel_path = SELECTED_PATH if not args.smoke else out_dir / "selected_params_SMOKE.json"
    with open(sel_path, "w", newline="\n") as f:
        json.dump(sel, f, indent=2)
    print("selected:", params_dict(p), "->", sel_path)

    df = evaluate(records, p, ALL_SYSTEMS)
    csv(df, out_dir / "per_query.csv", index=False)
    mrr = df.pivot_table(index="system", columns="group", values="rr", aggfunc="mean").reindex(list(ALL_SYSTEMS))
    csv(mrr, out_dir / "summary_mrr_by_group.csv")
    csv(df.pivot_table(index="system", columns="condition", values="rr", aggfunc="mean").reindex(list(ALL_SYSTEMS)),
        out_dir / "summary_mrr_by_condition.csv")
    csv(df.pivot_table(index="system", columns="group", values="hit@1", aggfunc="mean").reindex(list(ALL_SYSTEMS)),
        out_dir / "summary_acc1_by_group.csv")
    ls = linker_selection(tables["linked_joint"])
    csv(ls, out_dir / "linker_selection.csv", index=False)
    la = linking_accuracy(records, {r.linker: float(r.tau) for r in ls.itertuples()})
    csv(la, out_dir / "linking_accuracy.csv", index=False)
    tr = truncation_rates(records, lengths)
    csv(tr, out_dir / "truncation_rates.csv", index=False)
    ties = tie_diagnostics(df)
    csv(ties, out_dir / "tie_diagnostics.csv", index=False)
    pool = pool_analysis(records, p, proto["secondary_analyses"]["pool_size"], len(s.diseases))
    csv(pool, out_dir / "pool_analysis.csv", index=False)

    label = "SMOKE CHECK (mechanics only, not results)" if args.smoke else "DEVELOPMENT RESULTS (lexicon A, dev rows)"
    print(f"\n{label}: MRR by group\n{mrr.round(3).to_string()}")
    print(f"\nquery length (Dense objective)\n{tables['query_max_length'].round(4).to_string(index=False)}")
    print(f"\nlinker candidates at their own best (tau, beta)\n{ls.round(4).to_string(index=False)}")
    print(f"\nlinking accuracy per segment\n{la.round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()

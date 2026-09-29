"""Regression gate: the legacy compatibility mode must reproduce the stored
v3 per-query correctness (results/per_query_predictions_v3.csv) exactly,
for all 8 conditions, before any v4 experiment is run.

    venv\\Scripts\\python experiments\\00_verify_legacy_v3.py

Writes results/rebuild/legacy_v3_check.csv and exits non-zero on any mismatch.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hsr import RESULTS  # noqa: E402
from hsr import legacy_v3  # noqa: E402


def main():
    stored = pd.read_csv(RESULTS / "per_query_predictions_v3.csv")
    stored = {m: g.sort_values("query_index").correct.to_numpy() for m, g in stored.groupby("method")}
    got = legacy_v3.run()
    rows = []
    for m in legacy_v3.METHODS:
        diffs = int((got[m] != stored[m]).sum())
        rows.append({"method": m, "stored_correct": int(stored[m].sum()), "legacy_mode_correct": int(got[m].sum()),
                     "n": len(got[m]), "per_query_diffs": diffs})
    df = pd.DataFrame(rows)
    out = RESULTS / "rebuild" / "legacy_v3_check.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    ok = (df.per_query_diffs == 0).all()
    print("\nLEGACY REGRESSION:", "PASS (0 per-query differences)" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

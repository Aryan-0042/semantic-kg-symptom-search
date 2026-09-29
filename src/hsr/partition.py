"""Development / test row partition.

Per disease, DEV_PER_DISEASE rows are drawn (seeded) into the development
partition; the remaining rows form the test partition. Development work
(method design, tuning, debugging) may only query development rows and only
build indices from development rows. Test rows are queried once, in the final
evaluation, against one index built from the development rows only; no test
row ever enters an index.
"""
from collections import defaultdict

import numpy as np
import pandas as pd

DEV_PER_DISEASE = 3


def make_partition(rows, seed, dev_per_disease=DEV_PER_DISEASE):
    by_disease = defaultdict(list)
    for r in rows:
        by_disease[r.disease].append(r.row_id)
    rng = np.random.default_rng(seed)
    records = []
    for disease in sorted(by_disease):
        ids = sorted(by_disease[disease])
        if len(ids) <= dev_per_disease + 1:
            raise ValueError(f"{disease}: {len(ids)} rows cannot leave >=2 test rows")
        dev = set(rng.choice(ids, size=dev_per_disease, replace=False).tolist())
        records += [{"row_id": i, "disease": disease, "split": "dev" if i in dev else "test"} for i in ids]
    return pd.DataFrame(records).sort_values("row_id").reset_index(drop=True)


def check_partition(part, rows):
    assert len(part) == len(rows) and part.row_id.is_unique
    assert set(part.split) == {"dev", "test"}
    per = part.groupby(["disease", "split"]).size().unstack()
    assert (per["dev"] == DEV_PER_DISEASE).all(), "every disease needs the same number of dev rows"
    assert (per["test"] >= 2).all(), "every disease needs at least two test rows"
    return per

"""Compute per-query components for a regime.

For every query row r, a FoldIndex is built from index_rows_for(r), which
must not contain r (development: the other development rows; final test: the
development rows, which never include a test row). Consecutive rows with the
same index rows reuse one FoldIndex. Every query of r (all conditions) is
scored against that index. KG-linked vectors are precomputed for the tau grid.

encoders = {"query": {max_length: Encoder}, "doc": Encoder, "linkers": {name: Encoder}}
"""
from dataclasses import dataclass
import time

import numpy as np

from .data import normalize_symptom
from .queries import segments
from .scoring import FoldIndex, score_components


@dataclass
class Record:
    query: object          # queries.Query
    true_idx: int
    comp: object           # scoring.Components
    kg_linked: dict        # linker -> tau -> score vector
    link_diag: dict        # per-segment linking diagnostics (for linking accuracy)
    index_size: int        # number of index rows used
    n_query_tokens: int    # dense query tokens incl. special tokens, before truncation


def _link_diag(fi, comp, query):
    segs = segments(query.text)
    gold = [normalize_symptom(s) for s in query.gold_symptoms]
    aligned = len(segs) == len(gold)
    return {
        "segments": tuple(segs),
        "gold": tuple(gold) if aligned else None,
        "exact": tuple(frozenset(fi.exact_linker.link_clause(s)) for s in segs),
        "substring": tuple(frozenset(fi.substring_linker.link(s)) for s in segs),
        "semantic": {name: (ev.best_node, np.asarray(ev.best_sim, dtype=np.float32))
                     for name, ev in comp.links.items()},
    }


def compute_components(query_rows, index_rows_for, queries_for, corpus, diseases, encoders,
                       tau_grid, C_grid, log_every=20):
    d_idx = {d: i for i, d in enumerate(diseases)}
    tokenizer_enc = next(iter(encoders["query"].values()))
    records, t0 = [], time.time()
    fi, fi_key = None, None
    for k, r in enumerate(query_rows, 1):
        idx_rows = index_rows_for(r)
        if any(x.row_id == r.row_id for x in idx_rows):
            raise AssertionError(f"row {r.row_id} is in its own index")
        key = tuple(sorted(x.row_id for x in idx_rows))
        if key != fi_key:
            fi = FoldIndex(idx_rows, corpus, diseases, encoders["doc"], encoders["linkers"],
                           classifier_C=tuple(C_grid))
            fi_key = key
        for q in queries_for(r):
            comp = score_components(fi, q.text, encoders["query"])
            kgl = {n: {t: fi.kg.score(ev.linked(t)) for t in tau_grid} for n, ev in comp.links.items()}
            records.append(Record(q, d_idx[q.disease], comp, kgl, _link_diag(fi, comp, q), len(idx_rows),
                                  tokenizer_enc.n_tokens(q.text)))
        if log_every and k % log_every == 0:
            print(f"  {k}/{len(query_rows)} rows, {time.time() - t0:.0f}s", flush=True)
    for e in [encoders["doc"], *encoders["query"].values(), *encoders["linkers"].values()]:
        e.save()
    return records

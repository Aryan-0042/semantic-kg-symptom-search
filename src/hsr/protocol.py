"""Protocol loading, regimes and guards.

Data flow (PROTOCOL.md section 3):
  development  query rows = the 123 development rows; each is scored against
               an index of the other 122 development rows (leave-one-row-out:
               2 rows of its own disease, 3 of every other disease); lexicon A.
  final test   query rows = the 181 test rows; every one is scored against
               one fixed index of the 123 development rows. No test row ever
               enters an index, profile, KG, IDF or classifier.

The final-test regime refuses to run unless the protocol is frozen, the
working tree is clean, the selected parameters and lexicon B are committed
with matching hashes, and no final results exist yet (run once).
"""
import json
import subprocess

import pandas as pd
import yaml

from . import DATA, RESULTS, ROOT
from .data import diseases, load_corpus, load_rows
from .encoders import Encoder
from .lexicon import file_sha256, load_lexicon, validate_lexicon
from .data import symptom_vocabulary
from .partition import check_partition, make_partition
from .queries import build_row_queries

PROTOCOL_PATH = ROOT / "configs" / "protocol.yaml"
PARTITION_PATH = DATA / "protocol" / "partition.csv"
DISEASES_PATH = DATA / "protocol" / "diseases.csv"
SELECTED_PATH = ROOT / "configs" / "selected_params.json"
LEXICON_B_FREEZE = ROOT / "configs" / "lexicon_B_freeze.json"
FINAL_DIR = RESULTS / "rebuild" / "test"   # final evaluation output, written once


def load_protocol():
    with open(PROTOCOL_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def git_state():
    def run(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def load_partition(protocol, rows):
    """Create the partition on first use; afterwards verify it never changes."""
    part = make_partition(rows, protocol["seeds"]["partition"], protocol["data"]["dev_rows_per_disease"])
    if PARTITION_PATH.exists():
        stored = pd.read_csv(PARTITION_PATH)
        if not stored.equals(part):
            raise RuntimeError("data/protocol/partition.csv differs from the seeded partition; refusing to continue")
    else:
        PARTITION_PATH.parent.mkdir(parents=True, exist_ok=True)
        part.to_csv(PARTITION_PATH, index=False, lineterminator="\n")
    check_partition(part, rows)
    return part


def load_checked_lexicon(path, name, rows):
    lex = load_lexicon(ROOT / path, name)
    errors, warnings = validate_lexicon(lex, symptom_vocabulary(rows))
    if errors:
        raise ValueError(f"lexicon {name} invalid: " + "; ".join(errors))
    return lex, warnings


def build_encoders(protocol, linker_names, query_lengths):
    """{"query": {max_length: Encoder}, "doc": Encoder, "linkers": {name: Encoder}}."""
    m = protocol["models"]
    return {
        "query": {int(L): Encoder(m["dense_query"]["name"], int(L)) for L in query_lengths},
        "doc": Encoder(m["dense_doc"]["name"], m["dense_doc"]["max_length"]),
        "linkers": {n: Encoder(m["linker_candidates"][n]["name"], m["linker_candidates"][n]["max_length"],
                               pooling=m["linker_candidates"][n].get("pooling", "cls")) for n in linker_names},
    }


def check_diseases(diseases_):
    """Disease ids (the tie-break order) are frozen in data/protocol/diseases.csv."""
    df = pd.DataFrame({"disease_id": range(len(diseases_)), "disease": diseases_})
    if DISEASES_PATH.exists():
        if not pd.read_csv(DISEASES_PATH).equals(df):
            raise RuntimeError("data/protocol/diseases.csv differs from the sorted disease list; refusing to continue")
    else:
        DISEASES_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DISEASES_PATH, index=False, lineterminator="\n")


class Setup:
    def __init__(self):
        self.protocol = load_protocol()
        self.rows = load_rows(ROOT / self.protocol["data"]["rows"])
        self.corpus = load_corpus(ROOT / self.protocol["data"]["corpus"])
        self.diseases = diseases(self.rows)
        check_diseases(self.diseases)
        self.partition = load_partition(self.protocol, self.rows)
        split = dict(zip(self.partition.row_id, self.partition.split))
        self.dev_rows = [r for r in self.rows if split[r.row_id] == "dev"]
        self.test_rows = [r for r in self.rows if split[r.row_id] == "test"]
        self.seed = self.protocol["seeds"]["queries"]
        self.partial_ks = tuple(self.protocol["queries"]["partial_k"])

    # ---- development regime: only development rows exist ----
    def dev_regime(self, lexicon_A):
        dev = list(self.dev_rows)
        dev_ids = {r.row_id for r in dev}

        def index_rows_for(r):
            assert r.row_id in dev_ids, "development regime queried a non-development row"
            return [x for x in dev if x.row_id != r.row_id]

        def queries_for(r):
            return build_row_queries(r, self.seed, lexicon_A, self.partial_ks)

        return dev, index_rows_for, queries_for

    # ---- final-test regime: guarded, run once ----
    def test_regime(self, lexicon_A, lexicon_B, allow_unfrozen=False):
        if not allow_unfrozen:
            self.assert_final_ready(lexicon_B)
        dev = list(self.dev_rows)
        dev_ids = {r.row_id for r in dev}
        test_ids = {r.row_id for r in self.test_rows}

        def index_rows_for(r):
            assert r.row_id in test_ids and r.row_id not in dev_ids, "final regime queried a non-test row"
            return dev  # one fixed index: development rows only

        def queries_for(r):
            qs = build_row_queries(r, self.seed, lexicon_B, self.partial_ks)
            # diagnostic only: lay wording from the development lexicon A on the same test rows
            qs += [q for q in build_row_queries(r, self.seed, lexicon_A, self.partial_ks)
                   if q.condition.startswith("lay_")]
            return qs

        return list(self.test_rows), index_rows_for, queries_for

    def assert_final_ready(self, lexicon_B):
        problems = []
        if self.protocol.get("status") != "frozen":
            problems.append("configs/protocol.yaml is not frozen")
        if not SELECTED_PATH.exists():
            problems.append("configs/selected_params.json missing (run the development sweep first)")
        if not LEXICON_B_FREEZE.exists():
            problems.append("configs/lexicon_B_freeze.json missing (freeze lexicon B first)")
        else:
            frozen = json.load(open(LEXICON_B_FREEZE))["sha256"]
            if frozen != lexicon_B.sha256:
                problems.append("lexicon B hash differs from its frozen record")
        if git_state()["dirty"]:
            problems.append("git working tree is not clean")
        if FINAL_DIR.exists() and any(FINAL_DIR.iterdir()):
            problems.append("results/rebuild/test already exists: the final evaluation is run once")
        if problems:
            raise RuntimeError("final evaluation blocked:\n  - " + "\n  - ".join(problems))

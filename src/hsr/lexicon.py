"""Lay-paraphrase lexicons.

A lexicon is a CSV with one row per raw symptom key:

    symptom_key, variant_1, variant_2[, variant_3 ...][, notes]

Each variant is a standalone first-person lay sentence *without* its final
full stop (for example "My head is pounding"). Lexicon A is the development
lexicon; lexicon B is the independently authored, frozen test lexicon. No
tuning, method or resource may be derived from lexicon B.
"""
from dataclasses import dataclass
import hashlib
import re

import pandas as pd

from .data import normalize_symptom

# Characters the segmenter treats as clause boundaries; variants must not contain them.
FORBIDDEN = re.compile(r"[.,;:!?\n]")


@dataclass
class Lexicon:
    name: str
    variants: dict  # raw symptom key -> tuple of variant strings
    sha256: str
    path: str

    def n_variants(self, key):
        return len(self.variants[key])


def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_lexicon(path, name):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    vcols = [c for c in df.columns if c.startswith("variant_")]
    variants = {}
    for rec in df.itertuples(index=False):
        rec = rec._asdict()
        key = rec["symptom_key"].strip()
        vs = tuple(v.strip() for v in (rec[c] for c in vcols) if v and v.strip())
        variants[key] = vs
    return Lexicon(name=name, variants=variants, sha256=file_sha256(path), path=str(path))


def validate_lexicon(lex, vocabulary):
    """Return (errors, warnings). Errors block use of the lexicon."""
    errors, warnings = [], []
    missing = sorted(set(vocabulary) - set(lex.variants))
    extra = sorted(set(lex.variants) - set(vocabulary))
    if missing:
        errors.append(f"{len(missing)} symptom keys have no row: {missing[:10]}")
    if extra:
        errors.append(f"{len(extra)} unknown symptom keys: {extra[:10]}")
    seen = {}
    for key, vs in lex.variants.items():
        if not vs:
            errors.append(f"{key!r}: no variants")
        for v in vs:
            if FORBIDDEN.search(v):
                errors.append(f"{key!r}: variant contains a clause delimiter: {v!r}")
            if v.lower() == normalize_symptom(key):
                warnings.append(f"{key!r}: variant identical to the canonical name: {v!r}")
            if v.lower() in seen and seen[v.lower()] != key:
                warnings.append(f"variant {v!r} used for both {seen[v.lower()]!r} and {key!r}")
            seen[v.lower()] = key
    return errors, warnings

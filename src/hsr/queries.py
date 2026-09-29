"""Query construction for the wording x completeness conditions.

Every condition for a row is generated from that same row, so wording and
completeness are never confounded with which row was drawn:

    canonical_full          all of the row's symptoms, clinical names
    canonical_partial_k2    2 seeded symptoms of the row, clinical names
    canonical_partial_k3    3 seeded symptoms (the k2 pair plus one more)
    lay_full                all symptoms, one lexicon variant each
    lay_partial_k2/_k3      the same subsets, the same variants as lay_full

The k-subsets are nested prefixes of one seeded permutation of the row's
symptoms, so k3 adds exactly one symptom to k2. A k-subset exists only when
k < n (a partial query always drops at least one symptom): rows with three
symptoms have no k3 query. The analysis group 'partial' pools k2 and k3.

Queries are *composed* symptom by symptom, never produced by substituting
strings inside a finished sentence, so a paraphrase can never be rewritten
by a later rule (the v3 nested-substitution bug).
"""
from dataclasses import dataclass
import re

import numpy as np

from .data import normalize_symptom

PARTIAL_KS = (2, 3)
CONDITIONS = ("canonical_full", "canonical_partial_k2", "canonical_partial_k3",
              "lay_full", "lay_partial_k2", "lay_partial_k3")
GROUPS = ("canonical_full", "canonical_partial", "lay_full", "lay_partial")

# Independent random streams per row, so results do not depend on processing order.
STREAM_PARTIAL = 1
STREAM_VARIANT = 2

SEGMENT_SPLIT = re.compile(r"[.,;:!?\n]+")


def condition_group(condition):
    """'lay_partial_k3' -> 'lay_partial'; full conditions are their own group."""
    return re.sub(r"_k\d+$", "", condition)


@dataclass(frozen=True)
class Query:
    query_id: str
    row_id: int
    disease: str
    condition: str
    lexicon: str          # "none" for canonical conditions, else the lexicon name
    text: str
    gold_symptoms: tuple  # raw keys of the symptoms this query expresses, in order

    @property
    def group(self):
        return condition_group(self.condition)


def row_rng(seed, row_id, stream):
    return np.random.default_rng([seed, row_id, stream])


def partial_subsets(row, seed, ks=PARTIAL_KS):
    """{k: indices of the symptoms kept} for every k < n, nested across k.

    One seeded permutation per row; the k-subset is its first k entries,
    presented in the row's column order.
    """
    n = len(row.symptoms)
    perm = row_rng(seed, row.row_id, STREAM_PARTIAL).permutation(n)
    return {k: tuple(sorted(int(i) for i in perm[:k])) for k in ks if k < n}


def variant_choice(row, lexicon, seed):
    """Index of the lexicon variant used for each of the row's symptoms."""
    rng = row_rng(seed, row.row_id, STREAM_VARIANT)
    return tuple(int(rng.integers(lexicon.n_variants(s))) for s in row.symptoms)


def _cap(s):
    return s[:1].upper() + s[1:]


def canonical_text(symptom_keys):
    return _cap(", ".join(normalize_symptom(s) for s in symptom_keys)) + "."


def lay_text(phrases):
    return " ".join(_cap(p.strip()) + "." for p in phrases)


def build_row_queries(row, seed, lexicon=None, ks=PARTIAL_KS):
    """All conditions for one row. Lay conditions require a lexicon."""
    subsets = partial_subsets(row, seed, ks)
    full = row.symptoms

    def q(condition, tag, text, gold):
        suffix = f":{tag}" if tag != "none" else ""
        return Query(f"{row.row_id}:{condition}{suffix}", row.row_id, row.disease, condition, tag, text, gold)

    out = [q("canonical_full", "none", canonical_text(full), full)]
    for k, keep in subsets.items():
        part = tuple(full[i] for i in keep)
        out.append(q(f"canonical_partial_k{k}", "none", canonical_text(part), part))
    if lexicon is not None:
        choice = variant_choice(row, lexicon, seed)
        phrases = [lexicon.variants[s][c] for s, c in zip(full, choice)]
        out.append(q("lay_full", lexicon.name, lay_text(phrases), full))
        for k, keep in subsets.items():
            out.append(q(f"lay_partial_k{k}", lexicon.name, lay_text([phrases[i] for i in keep]),
                         tuple(full[i] for i in keep)))
    return out


def segments(text):
    """Clause segments: split on sentence and list punctuation, lower-cased."""
    return [s.strip().lower() for s in SEGMENT_SPLIT.split(text) if s.strip()]


def single_pass_substitute(text, mapping):
    """Replace every key of `mapping` in one left-to-right pass.

    Longest keys win, matches are whole-word, and replaced text is never
    scanned again. This is the fix for legacy dictionary paraphrasing, which
    applied str.replace sequentially and so rewrote earlier paraphrases.
    Returns (new_text, keys_replaced_in_order).
    """
    keys = sorted(mapping, key=len, reverse=True)
    pattern = re.compile(r"(?<![A-Za-z0-9])(" + "|".join(re.escape(k) for k in keys) + r")(?![A-Za-z0-9])")
    applied = []

    def repl(m):
        applied.append(m.group(1))
        return mapping[m.group(1)]

    return pattern.sub(repl, text), applied

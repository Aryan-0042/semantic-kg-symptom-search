"""Linking query text to knowledge-graph symptom nodes.

ExactLinker (primary baseline)
    Normalisation: lower-case; every run of non-alphanumeric characters
    becomes one space ('Toxic look (typhos)' -> 'toxic look typhos'); the
    same normalisation is applied to node names. Matching happens inside
    each clause (the query is first split on [.,;:!?\\n]), so a match never
    spans two clauses. Within a clause, a single left-to-right regex pass
    over all node names, longest name first at each position, bounded by
    token boundaries: whole words and phrases only, and matches never
    overlap, so 'internal itching' does not also produce 'itching' and
    'coughing' does not produce 'cough'.

SubstringLinker (historical v1-v3 rule; sensitivity / legacy comparison only)
    A node is linked when its name occurs anywhere in the lower-cased,
    whitespace-collapsed query, including inside longer words and inside
    longer node names ('coughing' -> 'cough'; 'internal itching' -> both
    'internal itching' and 'itching').

SemanticLinker (proposed intervention)
    linked(q) = exact(q) UNION { argmax_n cos(e(seg), e(n)) : seg in segments(q),
                                  max_n cos(e(seg), e(n)) >= tau }
    It only ever *adds* links to the exact baseline, so any difference between
    the two systems is attributable to the added semantic links. The node
    vocabulary is the fold's KG symptom set (names seen in the index rows).
"""
from dataclasses import dataclass
import re

import numpy as np

from .queries import SEGMENT_SPLIT, segments

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_text(s):
    """Lower-case; runs of non-alphanumerics -> one space; trimmed."""
    return " ".join(_NON_ALNUM.sub(" ", s.lower()).split())


class ExactLinker:
    def __init__(self, names):
        self.node_of = {}
        for n in sorted(set(names)):
            form = normalize_text(n)
            if form in self.node_of and self.node_of[form] != n:
                raise ValueError(f"node names {self.node_of[form]!r} and {n!r} normalise identically")
            self.node_of[form] = n
        forms = sorted(self.node_of, key=lambda f: (-len(f), f))
        self.pattern = re.compile(r"(?<![a-z0-9])(" + "|".join(re.escape(f) for f in forms) + r")(?![a-z0-9])")

    def link_clause(self, clause):
        return {self.node_of[m] for m in self.pattern.findall(normalize_text(clause))}

    def link(self, text):
        out = set()
        for clause in SEGMENT_SPLIT.split(text):
            out |= self.link_clause(clause)
        return out


class SubstringLinker:
    def __init__(self, names):
        self.names = sorted(set(names))

    def link(self, text):
        t = " ".join(text.lower().split())
        return {n for n in self.names if n in t}


@dataclass
class SegmentLinks:
    """Precomputed linking evidence for one query (tau-independent)."""
    exact: frozenset
    segments: tuple      # segment strings
    best_node: tuple     # best KG node per segment
    best_sim: np.ndarray  # cosine similarity of that node

    def linked(self, tau):
        sem = {n for n, s in zip(self.best_node, self.best_sim) if s >= tau}
        return set(self.exact) | sem


class SemanticLinker:
    def __init__(self, encoder, names):
        self.names = sorted(set(names))
        self.exact = ExactLinker(self.names)
        self.node_emb = encoder.encode(self.names)
        self.encoder = encoder

    def evidence(self, text):
        segs = tuple(segments(text))
        if segs:
            sims = self.encoder.encode(list(segs)) @ self.node_emb.T
            best = sims.argmax(axis=1)
            best_node = tuple(self.names[j] for j in best)
            best_sim = sims[np.arange(len(segs)), best]
        else:
            best_node, best_sim = (), np.zeros(0)
        return SegmentLinks(frozenset(self.exact.link(text)), segs, best_node, best_sim)

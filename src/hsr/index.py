"""Per-fold index: disease profiles, passage documents, BM25, knowledge graph.

Everything here is built from the *index rows* of a fold only (the labelled
rows the system is allowed to know). Passages come from the frozen guideline
corpus; each passage is tagged with one disease.

v4 design (see PROTOCOL.md):
  profile   all symptoms observed for the disease in the index rows, ordered
            by frequency (ties alphabetical), no truncation
  document  "<disease>. Symptoms: <profile>. <passage text>"
  BM25      rank_bm25 Okapi (k1=1.5, b=0.75), lower-cased alphanumeric tokens
  KG        disease -> set of symptom names in the index rows; symptom weight
            idf(s) = ln((D + 1) / (df(s) + 1)) + 1, D = number of diseases
"""
from collections import Counter, defaultdict
import re

import numpy as np
from rank_bm25 import BM25Okapi

from .data import normalize_symptom

TOKEN = re.compile(r"[a-z0-9]+")


def regex_tokens(text):
    return TOKEN.findall(text.lower())


def build_profiles(index_rows):
    """disease -> tuple of normalized symptom names, most frequent first."""
    counts = defaultdict(Counter)
    for r in index_rows:
        counts[r.disease].update(r.symptom_names)
    return {d: tuple(s for s, _ in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))) for d, c in counts.items()}


def build_documents(corpus, profiles):
    docs = []
    for rec in corpus.itertuples(index=False):
        disease = rec.disease_tag.strip()
        prof = ", ".join(profiles.get(disease, ()))
        docs.append(f"{disease}. Symptoms: {prof}. {rec.text}")
    return docs


class BM25Index:
    def __init__(self, docs, tokenizer=regex_tokens):
        self.tokenizer = tokenizer
        self.bm25 = BM25Okapi([tokenizer(d) for d in docs])

    def scores(self, query):
        return self.bm25.get_scores(self.tokenizer(query))


class KnowledgeGraph:
    """Disease-symptom graph with disease-level features precomputed.

    score(linked) returns, for every disease d, the IDF-weighted share of the
    linked query symptoms that d has:  sum_{s in Q & S_d} w(s) / sum_{s in Q} w(s).
    This is the same relevance formula as the legacy KG; only how query
    symptoms are found (linking) differs between the exact and semantic variants.
    """

    def __init__(self, index_rows, diseases):
        self.diseases = list(diseases)
        self.d_index = {d: i for i, d in enumerate(self.diseases)}
        sets = defaultdict(set)
        for r in index_rows:
            sets[r.disease].update(r.symptom_names)
        self.symptom_sets = {d: frozenset(sets.get(d, ())) for d in self.diseases}
        self.symptoms = sorted(set().union(*self.symptom_sets.values()))
        self.s_index = {s: j for j, s in enumerate(self.symptoms)}
        # incidence matrix: diseases x symptoms
        self.M = np.zeros((len(self.diseases), len(self.symptoms)), dtype=np.float64)
        for d, ss in self.symptom_sets.items():
            for s in ss:
                self.M[self.d_index[d], self.s_index[s]] = 1.0
        df = self.M.sum(axis=0)
        n_d = len(self.diseases)
        self.idf = np.log((n_d + 1) / (df + 1)) + 1.0

    def score(self, linked):
        """linked: iterable of normalized symptom names (unknown names ignored)."""
        cols = sorted({self.s_index[s] for s in linked if s in self.s_index})
        if not cols:
            return np.zeros(len(self.diseases))
        w = self.idf[cols]
        return (self.M[:, cols] @ w) / w.sum()

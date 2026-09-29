"""Per-disease component scores for one query against one fold index.

A FoldIndex is everything built from a fold's index rows. score_components()
returns every signal the systems need; systems.py then combines them. Nothing
in this module sees the query's own row: callers pass index rows that exclude
it (development: leave-one-row-out within development rows; final test: the
development rows only).

Passage -> disease aggregation is MaxP: a disease's score is the best score of
any of its passages. Unlike the legacy rank-discounted sum over the top 20
passages, this does not favour diseases that happen to have more passages.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .index import BM25Index, KnowledgeGraph, build_documents, build_profiles
from .linking import ExactLinker, SemanticLinker, SubstringLinker
from .queries import canonical_text


def maxp(doc_scores, doc_disease_idx, n_diseases):
    out = np.full(n_diseases, -np.inf)
    np.maximum.at(out, doc_disease_idx, doc_scores)
    return out


class FoldIndex:
    def __init__(self, index_rows, corpus, diseases, doc_encoder, linkers, classifier_C=(1.0,)):
        """linkers: dict name -> Encoder used for semantic linking.
        classifier_C: inverse regularisation strengths to fit (all kept, so
        the development sweep can select one without refitting)."""
        self.diseases = list(diseases)
        d_idx = {d: i for i, d in enumerate(self.diseases)}
        self.doc_disease = np.array([d_idx[t.strip()] for t in corpus.disease_tag])
        self.profiles = build_profiles(index_rows)
        self.docs = build_documents(corpus, self.profiles)
        self.bm25 = BM25Index(self.docs)
        self.doc_emb = doc_encoder.encode(self.docs)
        self.kg = KnowledgeGraph(index_rows, self.diseases)
        self.exact_linker = ExactLinker(self.kg.symptoms)
        self.substring_linker = SubstringLinker(self.kg.symptoms)
        self.linkers = {name: SemanticLinker(enc, self.kg.symptoms) for name, enc in linkers.items()}
        # Supervised baseline: TF-IDF + logistic regression on the canonical
        # text of the index rows (the only labelled examples available).
        texts = [canonical_text(r.symptoms) for r in index_rows]
        labels = [d_idx[r.disease] for r in index_rows]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"[a-z0-9]+")
        X = self.vectorizer.fit_transform([t.lower() for t in texts])
        self.classifiers = {C: LogisticRegression(C=C, max_iter=5000).fit(X, labels) for C in classifier_C}


@dataclass
class Components:
    bm25: np.ndarray
    dense: dict           # query max_length -> MaxP cosine scores
    kg_exact: np.ndarray
    kg_substring: np.ndarray  # historical substring matcher (sensitivity only)
    links: dict           # linker name -> SegmentLinks (tau-independent)
    classifier: dict      # C -> probability vector over diseases


def score_components(fi, query_text, query_encoders):
    """query_encoders: dict max_length -> Encoder (same model, different truncation)."""
    n = len(fi.diseases)
    bm25 = maxp(fi.bm25.scores(query_text), fi.doc_disease, n)
    dense = {L: maxp(fi.doc_emb @ enc.encode([query_text])[0], fi.doc_disease, n)
             for L, enc in query_encoders.items()}
    links = {name: lk.evidence(query_text) for name, lk in fi.linkers.items()}
    kg_exact = fi.kg.score(fi.exact_linker.link(query_text))
    kg_substring = fi.kg.score(fi.substring_linker.link(query_text))
    Xq = fi.vectorizer.transform([query_text.lower()])
    clf = {}
    for C, model in fi.classifiers.items():
        p = np.zeros(n)
        p[model.classes_] = model.predict_proba(Xq)[0]
        clf[C] = p
    return Components(bm25, dense, kg_exact, kg_substring, links, clf)

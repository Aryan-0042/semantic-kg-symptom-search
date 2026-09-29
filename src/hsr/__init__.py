"""hsr: symptom-to-condition retrieval, rebuild pipeline (v4).

The package is organised by pipeline stage:

    data       rows, symptom vocabulary, guideline corpus
    lexicon    lay-paraphrase lexicons (A = development, B = frozen test)
    queries    query construction for the canonical / lay / partial conditions
    partition  development / test row partition
    index      disease profiles, passage documents, BM25, knowledge graph
    encoders   Hugging Face bi-encoders with an on-disk embedding cache
    linking    exact (single-pass) and semantic symptom linking
    scoring    per-disease scorers: BM25, dense, KG, supervised classifier
    fusion     score fusion, candidate pools and reranking
    metrics    ranking metrics over the full disease list, deterministic tie-break
    systems    baselines, the proposed system and its matched controls
    tuning     development-only parameter selection
    analysis   summaries, primary tests, diagnostics, pool analysis
    stats      paired, disease-clustered statistics
    runner     per-query component computation for a regime
    protocol   protocol loading, regimes and final-evaluation guards
    legacy_v3  re-implementation of the v3 pipeline for regression checks
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
CACHE = ROOT / "cache"

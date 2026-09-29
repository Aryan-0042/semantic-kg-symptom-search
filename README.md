# Bridging the Lay-to-Clinical Vocabulary Gap

Code, data and complete results for the paper **"Bridging the Lay-to-Clinical Vocabulary Gap: Hybrid Retrieval with Semantic Knowledge Graph Linking for Symptom-to-Condition Search"** (Dhairya Panwar and Aryan Raosaheb Doule, Sharda University).

People describe symptoms in everyday words ("I cannot seem to catch my breath"), while medical sources use clinical terms ("breathlessness"). This project measures how much that mismatch costs lexical, dense, supervised and knowledge-graph retrieval over 41 conditions. It proposes a hybrid system (BM25 + MedCPT) whose symptom knowledge graph is reached by **semantic linking**: each clause of the query is linked to its nearest graph symptom with SapBERT.

## Main results (181 held-out records)

| System | Lay, full MRR | Lay, partial MRR |
|---|---|---|
| Hybrid (BM25 + MedCPT) | 0.457 | 0.348 |
| Hybrid + KG, exact symptom matching | 0.444 | 0.353 |
| **Hybrid + KG, semantic linking (proposed)** | **0.746** | **0.489** |

- With the pre-specified test lexicon (LLM-generated), the gain over exact matching on complete lay descriptions is **+0.302 MRR** (95% CI 0.198–0.400).
- With a second, hand-written lexicon it is **+0.328**.
- All four pre-specified comparisons are significant after Holm correction.

## Repository layout

| Path | Contents |
|---|---|
| `src/hsr/` | the pipeline package: data, queries, lexicons, index, linking, scoring, fusion, systems, tuning, metrics, statistics, analysis, and a legacy-regression module |
| `experiments/` | numbered entry points, run from the repository root (see below) |
| `tests/run_tests.py` | model-free integrity checks |
| `configs/` | frozen protocol (`protocol.yaml`), selected parameters, lexicon hash records |
| `data/` | symptom dataset, guideline corpus, lexicons A/B/C with provenance, the fixed dev/test partition |
| `results/rebuild/` | every development and test output: per-query ranks, summaries, statistics, logs, provenance |
| `audit/` | audit of earlier project versions and the historical test-overlap analysis |
| `legacy/` | superseded code, results, configs and drafts, kept for the record (see `legacy/README.md`) |
| `src/*.py`, `results/*.csv` (top level) | the historical v1–v3 pipeline and its results, examined in `RESEARCH_AUDIT.md` |

Key documents:
- [`PROTOCOL.md`](PROTOCOL.md): the frozen experimental protocol (v1.1) and its amendment log.
- [`results/rebuild/FINAL_PAPER_DISCLOSURES.md`](results/rebuild/FINAL_PAPER_DISCLOSURES.md): every deviation and transparency item.
- [`RELEASE_NOTES.md`](RELEASE_NOTES.md): how the study developed, phase by phase.
- [`RESEARCH_AUDIT.md`](RESEARCH_AUDIT.md): the audit of versions v0–v3 that led to the rebuild.
- [`DATA_SOURCES.md`](DATA_SOURCES.md): data provenance and licences.

## Setup

Python 3.11 is required. The published results were produced on Windows 11 with an NVIDIA RTX 3050 (CUDA 12.1).

```bash
python -m venv venv
# Windows: venv\Scripts\activate      Linux/macOS: source venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

- **CPU-only machines:** replace `torch==2.5.1+cu121` with `torch==2.5.1` in `requirements.txt`. The pipeline runs, but exact bitwise reproduction is only guaranteed with the pinned GPU environment.
- **Models:** three models download automatically from Hugging Face on first use (about 1.3 GB): `ncbi/MedCPT-Query-Encoder`, `ncbi/MedCPT-Article-Encoder` and `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`.

## Reproducing the paper

Run every command from the repository root. On Windows, use `venv\Scripts\python` in place of `python`.

**1. Integrity checks** (seconds):
```bash
python tests/run_tests.py
```

**2. Tables II, III and IV, compared against the published results** (a few minutes on a GPU):
```bash
python experiments/08_reproduce_paper.py
```
This re-runs the frozen final evaluation (the code of `experiments/03_final_eval.py`) with lexicon B and the lexicon A diagnostic, plus the second test with lexicon C (`experiments/06_human_lexicon_eval.py`). It writes to `results/reproduction/`, compares every per-query rank and every H1–H4 statistic with `results/rebuild/`, and ends with `REPRODUCTION: EXACT` when everything matches.

`03_final_eval.py` and `06_human_lexicon_eval.py` are the original **one-shot** scripts. Their guards refuse to run again because their outputs already exist, which is intended; use `08_reproduce_paper.py` to replicate.

**3. Latency** (hardware-dependent; the paper reports an RTX 3050 laptop GPU):
```bash
python experiments/05_latency.py --out results/reproduction/latency
```
Without `--out`, it overwrites the published `results/rebuild/test/latency_diagnostics.csv`.

**4. Post-hoc checks** (these rewrite their published outputs with identical content):
```bash
python experiments/04_posthoc_sensitivity.py   # without the 39 queries that name a condition
python experiments/07_reviewer_checks.py       # untouched records, two-way bootstrap, tie bounds, pool CIs, truncation
```

**5. Full pipeline from the start** (optional; development takes a few minutes):
```bash
python experiments/00_verify_legacy_v3.py   # the legacy v3 pipeline must report 0 differences
python experiments/01_prepare_protocol.py   # verifies the partition, disease ids and lexicon A; writes the dev queries
python experiments/02_dev_sweep.py          # development sweep; rewrites configs/selected_params.json identically
python experiments/02b_verify_selection.py  # re-derives every selected parameter from the saved tables
```

### Where each paper result comes from

| Paper element | Published file(s) | Produced by |
|---|---|---|
| Table II (linking example) | printed | `08_reproduce_paper.py` |
| Table III (main results) | `results/rebuild/test/per_query.csv`, `summary_mrr_by_group.csv`, `summary_acc1_by_group.csv` | `03_final_eval.py` (replicate with `08`) |
| Table IV (H1–H4) | `results/rebuild/test/primary_tests.csv`, `results/rebuild/test_human_lexicon/comparisons.csv` | `03`, `06` (replicate with `08`) |
| Table V (pool size) | `results/rebuild/test/pool_analysis.csv`, `results/rebuild/posthoc_reviewer_checks/pool_recall_and_ci.csv` | `03`, `07` |
| Linking accuracy, lexicon shift | `linking_accuracy.csv`, `diagnostic_lexicon_A_vs_B.csv` | `03`, `06` |
| Post-hoc robustness checks | `results/rebuild/test/sensitivity_*.csv`, `results/rebuild/posthoc_reviewer_checks/` | `04`, `07` |
| Latency | `results/rebuild/test/latency_diagnostics.csv` | `05` |
| Selected parameters | `configs/selected_params.json`, `results/rebuild/dev/` | `02`, `02b` |

## Data and licences

- **Symptom dataset:** *Disease Symptom Prediction* by Pranay Patil (Kaggle, CC BY-SA 4.0). The derived CSVs are shared under the same terms.
- **Guideline corpus** (`data/guideline_corpus.jsonl`, 104 short passages):
  - 101 passages come from MedlinePlus and 3 from the CDC; every passage records its source URL.
  - Of the MedlinePlus passages, 26 come from the A.D.A.M. Medical Encyclopedia pages (`medlineplus.gov/ency/`). That text is copyrighted by A.D.A.M., Inc. and is included here only as short excerpts for non-commercial research reproducibility, with attribution.
  - The remaining MedlinePlus and CDC text is U.S. government public-domain content.
- **Lexicons:** lexicon A was written by an AI assistant (Claude), lexicon B was generated by Gemini 1.5 Pro from a fixed prompt, and lexicon C was written by a co-author. See the provenance files in `data/lexicons/`.
- **Not included in this release:** third-party papers used for background reading, the out-of-scope safety test sets (which contain Reddit posts), and the 167 MB suicide-detection dataset. See `DATA_SOURCES.md`.

## Citation

See [`CITATION.cff`](CITATION.cff). The repository URL and the publication details will be added once available.

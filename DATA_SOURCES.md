# Data sources, licences and redistribution notes

This file records where each external input came from, under what terms, and whether it is versioned in git. Checksums are SHA-256 as of 2026-09-29.

## In scope for the paper

| File | Source | Licence | In git | SHA-256 |
|---|---|---|---|---|
| `data/raw/dataset.csv`, `Symptom-severity.csv`, `symptom_Description.csv`, `symptom_precaution.csv` | Kaggle *Disease Symptom Prediction* by Pranay Patil (`itachi9604/disease-symptom-description-dataset`, version 2); also on GitHub `itachi9604/Disease-Symptom-dataset` | **CC BY-SA 4.0** on Kaggle; the GitHub repository is marked **GPL-3.0** | yes | `dataset.csv`: `ebbd391c4ba4d64f57a00eb3d0a55f0ca9b920b0c5de6b9af4234e72519c9618` |
| `data/symptom_disease.csv` | Derived from `dataset.csv` by `src/prepare_dataset.py` + `src/generate_urgency_labels.py` | Derivative (CC BY-SA 4.0 share-alike) | yes | `c4f1c554d6eb6d09633246bf793e0ccda5f335f305c861d37cb452c48722b9a4` |
| `data/guideline_corpus.jsonl` | Passages scraped from MedlinePlus (101) and CDC (3) pages by `src/build_guideline_corpus.py` (Aug 2026). **Frozen artefact:** re-scraping will not reproduce it | MedlinePlus health-topic pages are public-domain US government content, except where marked as licensed from third parties (e.g. A.D.A.M. encyclopedia pages under `/ency/`), which are copyrighted. CDC content is generally public domain | yes | `e1b6395b1589ac5d8c0d5e395fcc05e69d41b30d868a1ad2505c156cef8fd740` |

## Out of scope for the paper (kept for history)

| File | Source | Licence | In git | Notes |
|---|---|---|---|---|
| `data/raw/Suicide_Detection.csv` (167 MB) | Kaggle *Suicide and Depression Detection* by Nikhileswar Komati (`nikhileswarkomati/suicide-watch`): posts from r/SuicideWatch, r/depression and r/teenagers collected via the Pushshift API | Listed as **CC BY-SA 4.0** by the uploader; the underlying posts are Reddit user content | **no** (`.gitignore`) | Excluded because it exceeds GitHub's 100 MB file limit, is sensitive mental-health text, is used only by `src/build_safety_testset.py` (out of scope), and can be re-downloaded from Kaggle. SHA-256: `1ec16b065a2fae36488de16e8abfc326586a576a25901e09467223e545ddd295` |
| `data/safety_testset.csv`, `data/safety_testset_for_review.csv` | Built by `src/build_safety_testset.py` / `finalize_safety_dataset.py`; include sampled Reddit posts from the file above | Inherits CC BY-SA 4.0 plus Reddit content concerns | yes (local history only) | **Review before any public release**: these contain Reddit suicide-watch posts. |

## Other third-party material

| Path | Notes |
|---|---|
| `reference_papers/` | Published papers converted to Markdown for study. They are third-party copyrighted and versioned only for local history. **Remove from any public release**, or publish only the citations. |
| MedCPT models (`ncbi/MedCPT-Query-Encoder`, `ncbi/MedCPT-Article-Encoder`) | Downloaded from Hugging Face into the user cache, not stored in the repository. |

## Before making the repository public

1. Remove or filter `reference_papers/` and `data/safety_testset*.csv` from the published history (or publish a fresh export rather than this history).
2. Keep attribution for the Kaggle symptom dataset (CC BY-SA 4.0) and note the share-alike terms for derived CSVs.
3. **Decision for the public release (Phase 8):**
   - The guideline corpus is published in full. Of its 104 passages, 26 are short excerpts from A.D.A.M. Medical Encyclopedia pages on MedlinePlus (`/ency/`), copyrighted by A.D.A.M., Inc. They are included, with attribution and source URLs, for non-commercial research reproducibility; the authors accepted responsibility for this choice.
   - The public export omits `reference_papers/` and `data/safety_testset*.csv`.

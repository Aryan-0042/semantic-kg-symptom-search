# Lexicon A (development) provenance

- **File:** `lexicon_A_dev.csv` (131 symptom keys × 2 variants)
- **Authored by:** Claude (AI assistant), on 2026-09-29, during the rebuild, before any v4 experiment had been run.
- **Role:** development only. It is used for tuning the linking threshold and fusion weights and for debugging, on development-partition rows. It must never be used to build, tune or evaluate against the test lexicon B.
- **Relation to earlier work:** written from scratch. It does not reuse `src/paraphrase_dictionary.py` (the v3 test instrument). The author of A had seen the v3 results; this is acceptable for a development resource and is disclosed here.
- **Style:** first-person lay sentences. Everyday words that coincide with clinical terms (for example "coughing") are kept where natural.

# Lexicon B-human (second test lexicon) provenance

- **File:** `lexicon_B_human.csv` (131 symptom keys × 2 variants)
- **Authored by:** a co-author of this study, by hand. According to the project lead, the co-author had not seen lexicon A, the Gemini-generated lexicon B, the code or any results.
- **Received and frozen:** 2026-09-29, after the one-shot final evaluation with the Gemini lexicon B (commit `33bd63d`). SHA-256 `ea4bf24d0bb0c5596a95daf487bfb584d7426b6d5655c97fd5bdf93fd0f576b1` (`configs/lexicon_B_human_freeze.json`).
- **Checks at freezing:** the protocol lexicon check passes (131 keys, 262 variants, no clause delimiters). 4–13 words per variant. No exact copies of any lexicon A or Gemini lexicon B variant; mean best content-word Jaccard overlap 0.20 with lexicon A and 0.18 with the Gemini lexicon. No variant names a disease.
- **Role:** a second, independently written test lexicon. The frozen system (configs/selected_params.json) is evaluated on it once, with no change to any parameter, by `experiments/06_human_lexicon_eval.py`. Its results are reported alongside the pre-specified results for the Gemini lexicon B, which remain the primary results.

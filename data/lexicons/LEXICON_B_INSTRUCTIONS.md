# Writing test lexicon B: instructions for the co-author

**Why this exists.** The study measures how well symptom-to-condition retrieval copes when patients describe symptoms in everyday language. The final test uses lay wordings that nobody on the method side has seen, so the result cannot have been tuned to them. That only works if lexicon B is written **independently**.

## Rules (please read before starting)

1. **Do not open** any of these while writing: `data/lexicons/lexicon_A_dev.csv`, `src/paraphrase_dictionary*.py`, anything under `results/`, `audit/`, `RESEARCH_AUDIT.md`, or any experiment output. Do not ask the other author how the system performs.
2. Work only from `lexicon_B_test_TEMPLATE.csv`: save your copy as `lexicon_B_test.csv` in this folder.
3. When you are done, record your name and the date in `LEXICON_B_PROVENANCE.md` (one line), and do not edit the file afterwards. The file is frozen by a git commit that records its SHA-256 hash.

## What to write

For every row, fill `variant_1` and `variant_2` with **two different ways a patient might describe that symptom** when typing into a symptom checker or messaging a doctor.

- Write in the **first person**, as a complete short sentence: *My head is pounding* / *I keep throwing up*.
- Use **everyday words** that a person without medical training would use. Where the clinical term is also the everyday word (for example, many people just say "I keep coughing"), it is fine to use it. Do not avoid words artificially; aim for how people really talk.
- Make the two variants genuinely different, not the same sentence with one word changed.
- 3–15 words per variant.
- **No commas, full stops, semicolons, colons, question marks or exclamation marks inside a variant.** The software splits sentences on punctuation. Apostrophes and hyphens are fine.
- Do not mention a disease name ("I think I have malaria").
- Some items are not really symptoms, for example `family_history`, `extra_marital_contacts`, `receiving_blood_transfusion` or `history_of_alcohol_consumption`. Write how a patient would naturally mention them ("My mum had the same thing").
- If you are unsure what a clinical term means (for example `scurring`, `dischromic patches`, `toxic look (typhos)` or `spotting urination`), look it up in a general medical reference, write your best lay description, and add a short remark in `notes`.

## Example (not from the real list)

| symptom_key | variant_1 | variant_2 |
|---|---|---|
| sore_throat | It hurts when I swallow | My throat feels scratchy and raw |

## Checking your file

Run `venv\Scripts\python experiments\01_prepare_protocol.py --check-lexicon data/lexicons/lexicon_B_test.csv`. This checks coverage and punctuation only; it reports no performance numbers.

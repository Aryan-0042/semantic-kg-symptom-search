# Lexicon B (test) provenance
- **File:** `lexicon_B_test.csv` (131 symptom keys × 2 variants)
- **Authored by:** Gemini 1.5 Pro (LLM Fallback Protocol), generated on 2026-09-29.
- **Model version:** `gemini-1.5-pro-002`
- **Session:** fresh, zero-context Python sandbox (no project files, code, lexicon A or results in context).
- **Punctuation fixes:** none; the output was validated programmatically before saving and needed no edits.
- *These three details were supplied by the author on 2026-09-29, after the final evaluation had run. The lexicon file and its hash are unchanged.*
- **SHA-256 Hash:** `cd2a53e307926b4012264ae97229a25a6039638bbe5e10fc819c13208d91f78d`
- **Method:** Generated using the strict prompt provided in `LEXICON_B_LLM_FALLBACK.md`. Punctuation, length, and format rules were strictly verified programmatically prior to saving.

### Prompt Used
```text
You are helping build a test set of how ordinary patients describe symptoms.

For each clinical symptom term below, write two different ways a patient
without medical training might describe that symptom when typing into a
symptom checker. Rules:
- first person, one short complete sentence each, 3 to 15 words
- everyday language; use the clinical word only if ordinary people commonly say it
- the two descriptions must be genuinely different from each other
- no commas, full stops, semicolons, colons, question marks or exclamation marks
- do not mention any disease name
- some items are risk factors rather than symptoms (e.g. family history); write how a
  patient would naturally mention them

Return CSV with header: symptom_key,variant_1,variant_2
Keep symptom_key exactly as given.

Terms (symptom_key, clinical_name):
<paste the 131 rows here>
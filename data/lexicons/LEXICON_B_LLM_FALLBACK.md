# Fallback for lexicon B: LLM generation with a fixed prompt

**Use this only if a human-authored lexicon B cannot be obtained.** If it is used, the paper must say so, and it must record the model name and version, the date, the exact prompt below, and the SHA-256 of the resulting file.

## Procedure

1. Pick one model (write its exact name and version in `LEXICON_B_PROVENANCE.md`). Use a fresh session with no project context.
2. Send the prompt below **once**, with the 131 `symptom_key` / `clinical_name` pairs from `lexicon_B_test_TEMPLATE.csv` pasted where indicated. Do not regenerate, cherry-pick or edit individual answers based on anything other than the formatting rules. Rows that break a formatting rule may be fixed only by removing the offending punctuation.
3. Save the output as `lexicon_B_test.csv`, run the lexicon check, and commit it before running the final evaluation.

## Prompt (fixed; do not edit)

```
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
```

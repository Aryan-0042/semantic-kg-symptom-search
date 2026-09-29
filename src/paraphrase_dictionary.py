"""
paraphrase_dictionary.py

PRE-REGISTERED before the v2 test set was built. EXTENDED once, before
any v3 accuracy result was observed, as a documented measurement-
instrument correction -- not a post-hoc tuning move.

=== CORRECTION LOG (v3) ===
An automated paraphrase-coverage check on the frozen v3 test set found
that 4 of 82 paraphrased rows (2 for Arthritis, 2 for Urinary tract
infection) received zero dictionary substitutions. Inspection of the
REAL underlying data confirmed this was not an unlucky draw: the
original dictionary covered ZERO of the 5 distinct symptom terms used
across all 6 real Arthritis rows, and ZERO of the 4 distinct terms used
across all 5 real Urinary tract infection rows. No alternative row for
either disease could have been drawn to avoid this gap.

Four terms were added to close it completely (verified against every
possible row for both diseases, not just the two rows already drawn,
since an initial 2-term proposal was checked and found to still miss
on at least one alternative row per disease):
  - "movement stiffness"   (Arthritis)
  - "painful walking"      (Arthritis)
  - "foul smell of urine"  (Urinary tract infection)
  - "burning micturition"  (Urinary tract infection)

This correction was made BEFORE any v3 retrieval accuracy result was
computed, and is disclosed here and in the paper's methodology section
for exactly that reason. No other terms were added or altered, and no
already-covered term was changed.

Maps canonical Symptom_1..17 vocabulary terms to a lay-language
paraphrase a patient might plausibly type instead. Coverage is partial
by design -- only symptoms with an unambiguous, natural lay equivalent
are included.
"""

PARAPHRASES = {
    "high fever": "a really high temperature",
    "mild fever": "a bit of a temperature",
    "fatigue": "feeling exhausted all the time",
    "cough": "a hacking cough that won't quit",
    "breathlessness": "can't catch my breath",
    "chest pain": "a tightness in my chest",
    "headache": "a pounding head",
    "muscle pain": "my muscles are so sore",
    "joint pain": "my joints ache",
    "nausea": "feeling sick to my stomach",
    "vomiting": "throwing up",
    "abdominal pain": "cramping in my stomach",
    "diarrhoea": "loose stools",
    "loss of appetite": "I don't feel like eating anything",
    "yellowish skin": "my skin looks yellow",
    "yellowing of eyes": "the whites of my eyes look yellow",
    "dark urine": "my urine is darker than usual",
    "weight loss": "I've been losing weight without trying",
    "sweating": "sweating a lot for no reason",
    "chills": "getting cold shivers",
    "malaise": "just feeling off / unwell",
    "dizziness": "feeling lightheaded",
    "blurred and distorted vision": "my eyesight has gone blurry",
    "itching": "my skin is really itchy",
    "skin rash": "a rash on my skin",
    "irritability": "I've been really irritable",
    "restlessness": "I can't sit still, feeling restless",
    "sunken eyes": "my eyes look sunken in",
    "dehydration": "I think I'm dehydrated",
    "swelled lymph nodes": "the glands in my neck are swollen",
    "continuous sneezing": "I can't stop sneezing",
    "runny nose": "my nose won't stop running",
    "congestion": "I'm all stuffed up",
    "throat irritation": "my throat feels raw",
    "watering from eyes": "my eyes keep watering",
    "loss of smell": "I can't smell anything",
    "internal itching": "an itching feeling inside",
    "stomach pain": "my stomach hurts",
    "acidity": "bad heartburn",
    "ulcers on tongue": "sores on my tongue",
    "weakness in limbs": "my arms and legs feel weak",
    "fast heart rate": "my heart is racing",
    "excessive hunger": "I'm hungry all the time",
    "increased appetite": "I can't stop eating",
    "polyuria": "I'm urinating way more than usual",
    "bloody stool": "blood in my stool",
    "irritation in anus": "irritation down there",
    "neck pain": "my neck hurts",
    "back pain": "my back is killing me",
    "weakness of one body side": "one side of my body feels weak",
    "altered sensorium": "I feel confused / not myself",
    "red spots over body": "red spots all over my body",
    "puffy face and eyes": "my face and eyes look puffy",
    "enlarged thyroid": "a swelling in my neck",
    "brittle nails": "my nails keep breaking",
    "swollen extremeties": "my hands and feet are swollen",
    "swollen legs": "my legs are swollen",
    "cold hands and feets": "my hands and feet are always cold",
    "mood swings": "my mood keeps swinging",
    "lethargy": "I have no energy at all",
    "sinus pressure": "pressure behind my face",
    "phlegm": "coughing up phlegm",
    "chest congestion": "my chest feels congested",

    # --- v3 correction (added before any v3 result was observed; see log above) ---
    "movement stiffness": "my joints feel stiff",
    "painful walking": "it hurts when I walk",
    "foul smell of urine": "my urine smells unusually strong",
    "burning micturition": "it burns when I urinate",
}


def paraphrase_query_text(query_text, rng, apply_probability=1.0):
    """
    Given a query string ('The patient reports <symptom>, <symptom>, ...'),
    replaces any matched canonical symptom substrings with their
    paraphrase, IF this query has been selected for paraphrasing upstream.

    Returns: (new_text, list_of_terms_actually_paraphrased)
    """
    if rng.random() > apply_probability:
        return query_text, []

    text = query_text
    applied = []
    for canonical in sorted(PARAPHRASES, key=len, reverse=True):
        if canonical in text:
            text = text.replace(canonical, PARAPHRASES[canonical])
            applied.append(canonical)
    return text, applied

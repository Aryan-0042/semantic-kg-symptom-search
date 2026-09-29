"""
paraphrase_dictionary.py

PRE-REGISTERED before the new test set is drawn or evaluated, and before
any results from the new architecture exist. This file is committed as
part of the test-set-construction protocol, not derived from it.

Maps canonical Symptom_1..17 vocabulary terms (as they appear in the
Kaggle dataset and in the KG's symptom nodes) to a lay-language
paraphrase a patient might plausibly type instead. Coverage is partial
by design -- only symptoms with an unambiguous, natural lay equivalent
are included; terms without a clean paraphrase are left as-is rather
than forcing an awkward substitution, and the test-set builder records
exactly which terms in each query were and weren't paraphrased.

This dictionary intentionally does NOT touch the corpus text, the KG
symptom nodes, or the retrieval-text construction -- only the query
strings shown to the retrieval system at evaluation time. That asymmetry
(patient-phrased query vs. clinically-phrased knowledge base) is the
condition Option A's dense-retrieval stage exists to handle, and is
exactly what the canonical-vocabulary queries used in the original 82
frozen test queries did not exercise.
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
}


def paraphrase_query_text(query_text, rng, apply_probability=1.0):
    """
    Given a query string built by build_queries_from_split() (i.e.
    'The patient reports <symptom>, <symptom>, ...'), replaces any
    matched canonical symptom substrings with their paraphrase, IF this
    query has been selected for paraphrasing upstream (apply_probability
    is a knob for partial-coverage experiments; the test-set builder
    below uses 0.0/1.0 per query, decided once, not per symptom).

    Returns: (new_text, list_of_terms_actually_paraphrased)
    """
    if rng.random() > apply_probability:
        return query_text, []

    text = query_text
    applied = []
    # Longest terms first, so "high fever" matches before a hypothetical
    # shorter overlapping key would.
    for canonical in sorted(PARAPHRASES, key=len, reverse=True):
        if canonical in text:
            text = text.replace(canonical, PARAPHRASES[canonical])
            applied.append(canonical)
    return text, applied

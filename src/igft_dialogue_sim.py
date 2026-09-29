"""
igft_dialogue_sim.py -- AUDITED VERSION

CHANGE FROM ORIGINAL (Phase 2): the symptom-disease prior/conditional
tables were previously built ONCE from the full dataset, before the
trial loop, meaning every simulated patient's own row was included in
the very tables used to diagnose them (leave-NONE-out). This has been
fixed to true leave-one-out: for each trial, the tables are built
excluding that trial's sampled patient row. Everything else --
simulation logic, policies (fixed/random/igft), turn budget, confidence
threshold, output schema -- is unchanged from the original.

Only used for symptom_disease.csv's existing Disease/Symptom_1..17
columns; no dependency on the retrieval pipeline's splits.
"""
import numpy as np
import pandas as pd
import random

TAU = 0.65
MAX_TURNS = 12
N_TRIALS = 300


def load_symptom_disease(path="data/symptom_disease.csv"):
    df = pd.read_csv(path)
    symptom_cols = [c for c in df.columns if c.lower().startswith("symptom")]
    df[symptom_cols] = df[symptom_cols].fillna("").apply(lambda c: c.str.strip().str.lower())
    df["symptom_set"] = df[symptom_cols].apply(lambda r: set(s for s in r if s), axis=1)
    return df[["Disease", "symptom_set"]]


def build_symptom_disease_prior(df):
    """Unchanged logic from the original -- now called per-trial on a
    leave-one-out subset instead of once on the full dataframe."""
    diseases = sorted(df["Disease"].unique())
    all_symptoms = sorted(set().union(*df["symptom_set"])) if len(df) else []
    prior = {d: (df["Disease"] == d).mean() for d in diseases}
    cond = {d: {} for d in diseases}
    for d in diseases:
        rows = df[df["Disease"] == d]["symptom_set"]
        n = len(rows)
        for s in all_symptoms:
            present = sum(1 for r in rows if s in r)
            cond[d][s] = (present + 1) / (n + 2)
    return diseases, all_symptoms, prior, cond


def posterior(prior, cond, disease_list, revealed_present, revealed_absent):
    logp = {}
    for d in disease_list:
        lp = np.log(prior[d] + 1e-12)
        for s in revealed_present:
            lp += np.log(cond[d].get(s, 1e-6))
        for s in revealed_absent:
            lp += np.log(1 - cond[d].get(s, 1e-6) + 1e-12)
        logp[d] = lp
    m = max(logp.values())
    exp_p = {d: np.exp(v - m) for d, v in logp.items()}
    z = sum(exp_p.values())
    return {d: v / z for d, v in exp_p.items()}


def entropy(p_dict):
    ps = np.array(list(p_dict.values()))
    ps = ps[ps > 0]
    return -np.sum(ps * np.log2(ps))


def expected_info_gain(candidate_symptom, disease_list, prior, cond, post, revealed_present, revealed_absent):
    p_present = sum(post[d] * cond[d].get(candidate_symptom, 1e-6) for d in disease_list)
    p_absent = 1 - p_present
    h_now = entropy(post)
    post_if_present = posterior(prior, cond, disease_list, revealed_present | {candidate_symptom}, revealed_absent)
    post_if_absent = posterior(prior, cond, disease_list, revealed_present, revealed_absent | {candidate_symptom})
    h_present = entropy(post_if_present)
    h_absent = entropy(post_if_absent)
    return h_now - (p_present * h_present + p_absent * h_absent)


def simulate_patient(true_disease, true_symptoms, all_symptoms, disease_list, prior, cond, policy):
    revealed_present, revealed_absent = set(), set()
    asked = set()
    for turn in range(1, MAX_TURNS + 1):
        post = posterior(prior, cond, disease_list, revealed_present, revealed_absent)
        top_d, top_p = max(post.items(), key=lambda x: x[1])
        if top_p >= TAU and top_d == true_disease:
            return turn, True
        if top_p >= TAU and top_d != true_disease:
            return turn, False

        remaining = [s for s in all_symptoms if s not in asked]
        if not remaining:
            return MAX_TURNS, (top_d == true_disease)

        if policy == "random":
            q = random.choice(remaining)
        elif policy == "fixed":
            q = remaining[0]
        elif policy == "igft":
            gains = [(s, expected_info_gain(s, disease_list, prior, cond, post, revealed_present, revealed_absent))
                     for s in remaining[:40]]
            q = max(gains, key=lambda x: x[1])[0]
        else:
            raise ValueError(policy)

        asked.add(q)
        if q in true_symptoms:
            revealed_present.add(q)
        else:
            revealed_absent.add(q)

    post = posterior(prior, cond, disease_list, revealed_present, revealed_absent)
    top_d, _ = max(post.items(), key=lambda x: x[1])
    return MAX_TURNS, (top_d == true_disease)


def main():
    df = load_symptom_disease()

    rows = []
    for policy in ["fixed", "random", "igft"]:
        for _ in range(N_TRIALS):
            sampled_idx = df.sample(1).index[0]
            sampled = df.loc[sampled_idx]

            # LEAVE-ONE-OUT FIX: exclude this trial's own patient row from
            # the tables used to diagnose it. This is the only behavioral
            # change from the original script.
            loo_df = df.drop(index=sampled_idx)
            disease_list, all_symptoms, prior, cond = build_symptom_disease_prior(loo_df)

            # Guard: if the sampled disease no longer has ANY remaining rows
            # after leave-one-out (only possible for a disease with exactly
            # 1 total row in the dataset), skip -- it cannot be fairly
            # evaluated leave-one-out at all, and silently keeping it in
            # would reintroduce the exact leak this fix removes.
            if sampled["Disease"] not in disease_list or (df["Disease"] == sampled["Disease"]).sum() < 2:
                continue

            turns, correct = simulate_patient(
                sampled["Disease"], sampled["symptom_set"], all_symptoms, disease_list, prior, cond, policy
            )
            rows.append({"policy": policy, "turns": turns, "correct": correct})

    out = pd.DataFrame(rows)
    summary = out.groupby("policy").agg(
        mean_turns=("turns", "mean"),
        std_turns=("turns", "std"),
        accuracy=("correct", "mean"),
        n_trials=("turns", "count"),
    ).reset_index()

    import os
    os.makedirs("results", exist_ok=True)
    out.to_csv("results/igft_raw.csv", index=False)
    summary.to_csv("results/igft_summary.csv", index=False)
    print(summary)
    print("\nNote: prior/conditional tables were rebuilt per-trial with the sampled")
    print("patient's own row excluded (leave-one-out). Any disease with fewer than")
    print("2 total rows in the dataset was skipped for this simulation, since it")
    print("cannot be evaluated leave-one-out at all.")


if __name__ == "__main__":
    main()

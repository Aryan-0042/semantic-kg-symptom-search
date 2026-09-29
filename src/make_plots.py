"""
make_plots.py -- LOCKED, FINAL VERSION for Phase 2.

CHANGE FROM PREVIOUS VERSION: fig_latency() had a real bug -- passing a
full-length yerr array into sns.barplot() on already-aggregated data
raises a shape-mismatch ValueError, because seaborn processes each
category as its own internal bar with length-1 y, not length-2. Fixed
by plotting directly with matplotlib's plt.bar(), which takes explicit
x positions and a yerr array with no ambiguity about shape.

ALSO CHANGED: fig_latency()'s title and this file's docstring now state
plainly that latency_summary.csv is a SIMULATED placeholder benchmark
(model_forward() in latency_benchmark.py is time.sleep(0.01), not your
retrieval pipeline) and must not be read as a measurement of edge vs.
cloud deployment cost for the six retrieval methods in Table II. This
figure is now saved as figs/latency_illustrative.pdf (renamed from
figs/latency.pdf) so it cannot be silently mistaken for a Results figure
by filename alone -- use it in Discussion/Future Work only, if at all,
not in the Results section.

Nothing else in this file changed from the previously verified version:
fig_baseline_bar(), fig_subset_breakdown(), fig_ablation_bar(),
fig_training_curves(), fig_confusion_matrix(), fig_igft_turns() are
byte-identical to the version already checked against your real
retrieval_comparison_v2.csv / subset_breakdown_v2.csv /
final_test_confidence_intervals_v2.csv data.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
os.makedirs("figs", exist_ok=True)


def fig_baseline_bar():
    df = pd.read_csv("results/retrieval_comparison_v2.csv")
    ci = pd.read_csv("results/final_test_confidence_intervals_v2.csv")

    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    melted = df.melt(id_vars="Method", value_vars=metrics, var_name="Metric", value_name="Score")

    plt.figure(figsize=(8, 4.2))
    ax = sns.barplot(data=melted, x="Method", y="Score", hue="Metric")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Score (%)")
    plt.title("Retrieval Comparison — Frozen Confirmatory Test Set (n=82)")

    method_order = df["Method"].tolist()
    n_metrics = len(metrics)
    bar_width = 0.8 / n_metrics
    acc_metric_idx = metrics.index("Accuracy")
    for i, method in enumerate(method_order):
        row = ci[ci["Method"] == method]
        if row.empty:
            continue
        acc = row["Accuracy"].iloc[0]
        lo = row["Wilson_95CI_low"].iloc[0]
        hi = row["Wilson_95CI_high"].iloc[0]
        x = i - 0.4 + bar_width * acc_metric_idx + bar_width / 2
        ax.errorbar(x, acc, yerr=[[acc - lo], [hi - acc]], fmt="none",
                     ecolor="black", elinewidth=1.2, capsize=3)

    plt.tight_layout()
    plt.savefig("figs/baseline_comparison.pdf")
    plt.close()


def fig_subset_breakdown():
    df = pd.read_csv("results/subset_breakdown_v2.csv")
    plt.figure(figsize=(8, 4.2))
    sns.barplot(data=df, x="Method", y="Accuracy", hue="Subset",
                hue_order=["canonical", "paraphrased"])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 105)
    plt.title("Canonical vs. Paraphrased Query Accuracy by Method (n=41 each)")
    plt.tight_layout()
    plt.savefig("figs/subset_breakdown.pdf")
    plt.close()


def fig_ablation_bar():
    # Unchanged -- original linear-CRFS ablation, not part of the v2 round.
    df = pd.read_csv("results/ablation.csv")
    plt.figure(figsize=(6.5, 4))
    sns.barplot(data=df, x="Method", y="F1", color="#4C72B0")
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("F1 Score (%)")
    plt.title("Ablation Study: Component Contribution (Linear CRFS)")
    plt.tight_layout()
    plt.savefig("figs/ablation.pdf")
    plt.close()


def fig_training_curves():
    df = pd.read_csv("results/triage_training_history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(df["epoch"], df["train_loss"], label="Train")
    axes[0].plot(df["epoch"], df["val_loss"], label="Validation")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].legend(); axes[0].set_title("Loss")
    axes[1].plot(df["epoch"], df["train_acc"], label="Train")
    axes[1].plot(df["epoch"], df["val_acc"], label="Validation")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy"); axes[1].legend(); axes[1].set_title("Accuracy")
    plt.suptitle("Triage Training Dynamics (single 80/20 split — see 5-fold CV for headline accuracy)")
    plt.tight_layout()
    plt.savefig("figs/training_curves.pdf")
    plt.close()


def fig_confusion_matrix():
    cm = np.loadtxt("results/triage_confusion_matrix.csv", delimiter=",")
    labels = ["Routine", "Urgent", "Emergency"][:cm.shape[0]]
    plt.figure(figsize=(4.2, 3.8))
    sns.heatmap(cm, annot=True, fmt=".0f", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title("Triage Confusion Matrix (single split — supplementary)")
    plt.tight_layout()
    plt.savefig("figs/confusion_matrix.pdf")
    plt.close()


def fig_igft_turns():
    df = pd.read_csv("results/igft_raw.csv")
    plt.figure(figsize=(5.5, 4))
    sns.boxplot(data=df, x="policy", y="turns")
    plt.xlabel("Questioning Policy"); plt.ylabel("Turns to Diagnosis")
    plt.title("Diagnostic Efficiency by Questioning Policy (leave-one-out corrected)")
    plt.tight_layout()
    plt.savefig("figs/igft_turns.pdf")
    plt.close()


def fig_latency_illustrative():
    """
    FIXED (yerr shape bug) and RELABELED. This benchmark's model_forward()
    is a time.sleep(0.01) placeholder, not your retrieval pipeline -- it
    measures nothing about BM25, MedCPT, or the KG. Saved under a
    filename and title that make this unambiguous. Do not cite these
    numbers as a measured result; use only as an illustrative deployment-
    cost sketch in Discussion/Future Work, if at all.
    """
    df = pd.read_csv("results/latency_summary.csv")
    x = np.arange(len(df))

    plt.figure(figsize=(5, 4))
    plt.bar(x, df["mean"], yerr=df["std"], capsize=4,
            color=["#55A868" if m == "edge_sim" else "#C44E52" for m in df["mode"]])
    plt.xticks(x, df["mode"])
    plt.ylabel("Mean Latency (ms)")
    plt.title("Edge vs. Cloud Latency\n(SIMULATED PLACEHOLDER — not a measured result)")
    plt.figtext(0.5, -0.02,
                "model_forward() is a fixed time.sleep(0.01) placeholder; this figure does not\n"
                "measure BM25, MedCPT, or KG inference. Illustrative only.",
                ha="center", fontsize=7, style="italic")
    plt.tight_layout()
    plt.savefig("figs/latency_illustrative.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    fig_baseline_bar()
    fig_subset_breakdown()
    fig_ablation_bar()
    fig_training_curves()
    fig_confusion_matrix()
    fig_igft_turns()
    fig_latency_illustrative()
    print("All figures written to figs/*.pdf")
    print("\nNOTE: figs/latency_illustrative.pdf is a simulated placeholder, not a real")
    print("measurement -- do not use it in the Results section (see docstring/caption).")

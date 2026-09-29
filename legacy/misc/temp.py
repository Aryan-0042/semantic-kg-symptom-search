import os

# The files you want to combine
# The files you want to combine
files_to_read = [
    # Retrieval Evaluation (Phase 2 + Option A)
    # "results/retrieval_comparison_v2.csv",
    # "results/subset_breakdown_v2.csv",
    # "results/significance_tests_v2.csv",
    # "results/final_test_confidence_intervals_v2.csv",
    # "results/per_query_predictions_v2.csv",

    # # Latency Benchmark
    # "results/latency_raw.csv",
    # "results/latency_all_methods_raw.csv",
    # "results/latency_all_methods_summary.csv",
    # "results/latency_method_comparison.csv"
    # "results/paraphrase_coverage_v3.csv",
    # "results/dev_val_coverage_v3.csv"
    # "results/rrf_shortlist_tuning_report_v3.csv",
    # "results/frozen_config_rrf_v3.json"
    # "results/rrf_dev_val_rank_diagnostic_v3.csv",
    # "results/rrf_dev_val_rank_distribution_v3.csv"

    # "results/retrieval_comparison_v3.csv",
    # "results/ablation_attribution_v3.csv",
    # "results/final_test_confidence_intervals_v3.csv",
    # "results/subset_breakdown_v3.csv",
    # "results/significance_tests_v3.csv",
    # "results/bootstrap_ci_v3.csv",
    # "results/per_query_predictions_v3.csv",
    # "results/per_disease_agreement_v3.csv"
    # "results/recency_variance_check_v3.csv"

    "results\master_verification_v3\master_bootstrap_v3.csv", 
    "results\master_verification_v3\master_confidence_intervals_v3.csv", 
    "results\master_verification_v3\master_main_claim_check_v3.csv", 
    "results\master_verification_v3\master_per_query_v3.csv", 
    "results\master_verification_v3\master_significance_v3.csv", 
    "results\master_verification_v3\master_subset_breakdown_v3.csv", 
    "results\master_verification_v3\master_subset_claim_check_v3.csv", 
    "results\master_verification_v3\master_summary_v3.csv", 
    "results\master_verification_v3\per_disease_results_v3.csv"
]

# The output text file in the root folder
output_file = "combined_results.txt"

def combine_files(file_list, output_path):
    # Open the output file in write mode
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for file_path in file_list:
            # Write the file name clearly as a header
            outfile.write(f"========== {file_path} ==========\n")
            
            # Check if the file exists to prevent crashing
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        content = infile.read()
                        outfile.write(content)
                        
                        # Ensure there's a newline at the end of the content
                        if not content.endswith('\n'):
                            outfile.write('\n')
                except Exception as e:
                    outfile.write(f"[Error reading file: {e}]\n")
            else:
                outfile.write("[File not found]\n")
            
            # Add some blank space before the next file
            outfile.write("\n\n")
            
    print(f"Done! Combined data saved to '{output_path}'.")

if __name__ == "__main__":
    combine_files(files_to_read, output_file)



# import pandas as pd
# from statsmodels.stats.contingency_tables import mcnemar

# df = pd.read_csv("results/per_query_predictions_v2.csv")

# def mcnemar_subset(method_a, method_b, subset_only=None):
#     d = df if subset_only is None else df[df["paraphrased"] == subset_only]
#     a = d[d.method == method_a].sort_values("query_index")["correct"].to_numpy()
#     b = d[d.method == method_b].sort_values("query_index")["correct"].to_numpy()
#     n01 = int(((a == 0) & (b == 1)).sum())
#     n10 = int(((a == 1) & (b == 0)).sum())
#     if n01 == 0 and n10 == 0:
#         return n01, n10, 1.0
#     p = mcnemar([[0, n01], [n10, 0]], exact=True).pvalue
#     return n01, n10, p

# for a, b in [("Hybrid+KG (no recency)", "Proposed (RRF+KG+Recency)"),
#              ("Hybrid", "Proposed (RRF+KG+Recency)")]:
#     n01, n10, p = mcnemar_subset(a, b, subset_only=True)
#     print(f"[paraphrased only] {a} vs {b}: n01={n01} n10={n10} p={p:.4f}")


# import pandas as pd
# df = pd.read_csv("data/symptom_disease.csv")
# subset = df[df["Disease"].isin(["Arthritis", "Urinary tract infection"])]
# subset.to_csv("arthritis_uti_subset.csv", index=False)
# print(subset.to_string())


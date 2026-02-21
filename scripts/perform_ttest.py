import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# --- MAPPING CONSTANT REMOVED FOR BREVITY (Keep your existing mapping) ---
MAPPING = {
    # ... (Paste your MAPPING dictionary here) ...
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare two graph clustering methods using paired t-tests with Holm-Bonferroni correction."
    )
    parser.add_argument(
        "--input-csv", type=str, required=True, help="Path to input CSV"
    )
    parser.add_argument(
        "--method-a", type=str, required=True, help="First method name"
    )
    parser.add_argument(
        "--method-b", type=str, required=True, help="Second method name"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["ami", "ari", "nmi", "f1_score", "precision", "recall"],
        help="List of metrics to compare",
    )
    parser.add_argument(
        "--output-file", type=str, default="comparison_report.txt", help="Output path"
    )
    parser.add_argument(
        "--network-list", type=str, default=None, help="Optional network filter list"
    )
    return parser.parse_args()


def get_paired_data(df, method_a, method_b, metrics, network_list):
    """(Same filtering logic as before - ensures matched pairs)"""
    if network_list:
        df = df[df["network_id"].isin(network_list)].copy()

    # Map if raw keys passed
    method_a = MAPPING.get(method_a, method_a)
    method_b = MAPPING.get(method_b, method_b)

    # Filter for methods
    df = df[df["Method"].isin([method_a, method_b])].copy()
    
    # Ensure mapping in column
    df["Method"] = df["Method"].apply(lambda x: MAPPING.get(x, x))

    # Intersection check
    valid_nets = []
    for net_id, group in df.groupby("network_id"):
        if group["Method"].nunique() == 2 and not group[metrics].isna().any().any():
            valid_nets.append(net_id)

    df_clean = df[df["network_id"].isin(valid_nets)].copy()
    return df_clean, method_a, method_b


def main():
    args = parse_args()
    
    # Load Data
    df = pd.read_csv(args.input_csv)
    
    # Load Network Filter
    network_list = None
    if args.network_list and Path(args.network_list).exists():
        with open(args.network_list) as f:
            network_list = [l.strip() for l in f if l.strip()]

    # Get Data
    df_clean, method_a, method_b = get_paired_data(
        df, args.method_a, args.method_b, args.metrics, network_list
    )

    if df_clean.empty:
        logger.error("No valid paired data found.")
        return

    # --- Statistical Testing ---
    
    raw_p_values = []
    metric_results = []

    # 1. Run raw t-tests for every metric
    for metric in args.metrics:
        pivoted = df_clean.pivot(index="network_id", columns="Method", values=metric)
        data_a = pivoted[method_a]
        data_b = pivoted[method_b]

        mean_a = data_a.mean()
        mean_b = data_b.mean()
        
        stat, p_val = ttest_rel(data_a, data_b)
        
        raw_p_values.append(p_val)
        metric_results.append({
            "metric": metric,
            "mean_a": mean_a,
            "mean_b": mean_b,
            "diff": mean_a - mean_b,
            "t_stat": stat,
            "raw_p": p_val
        })

    # 2. Apply Holm-Bonferroni Correction
    # multipletests returns: reject (bool array), corrected_p (array), ...
    reject_array, corrected_p_values, _, _ = multipletests(
        raw_p_values, 
        alpha=0.05, 
        method='holm' # 'holm' is the Holm-Bonferroni method
    )

    # 3. Generate Report
    with open(args.output_file, "w") as f:
        f.write(f"Statistical Comparison: {method_a} vs {method_b}\n")
        f.write(f"Networks (N): {df_clean['network_id'].nunique()}\n")
        f.write(f"Correction Method: Holm-Bonferroni (Step-down)\n")
        f.write(f"Alpha: 0.05\n")
        f.write("=" * 80 + "\n\n")

        # Zip results with the correction outcome
        for i, res in enumerate(metric_results):
            metric = res['metric']
            is_sig = reject_array[i]
            adj_p = corrected_p_values[i]
            
            sig_str = "YES" if is_sig else "NO"
            
            conclusion = "Inconclusive"
            if is_sig:
                if res['mean_a'] > res['mean_b']:
                    conclusion = f"BETTER: {method_a}"
                else:
                    conclusion = f"BETTER: {method_b}"

            f.write(f"Metric: {metric}\n")
            f.write(f"  Means:      {method_a}={res['mean_a']:.4f}, {method_b}={res['mean_b']:.4f}\n")
            f.write(f"  Raw p-val:  {res['raw_p']:.4e}\n")
            f.write(f"  Adj p-val:  {adj_p:.4e} (Holm)\n")
            f.write(f"  Significant? {sig_str}\n")
            f.write(f"  Conclusion: {conclusion}\n")
            f.write("-" * 50 + "\n")
            
    logger.info(f"Report saved to {args.output_file}")

if __name__ == "__main__":
    main()
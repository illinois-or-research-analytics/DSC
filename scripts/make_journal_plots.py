import sys
import logging
import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# --- Constants ---

MAPPING = {
    # DSC methods
    "fista-int": "DSC-FISTA(int)",
    "fista-int+wcc": "DSC-FISTA(int)+WCC",
    "fista-int-iter": "DSC-FISTA(int)-Iter",
    "fista-int-iter+wcc": "DSC-FISTA(int)-Iter+WCC",
    "fista-frac-iter": "DSC-FISTA-Iter",
    "fista-frac-iter+wcc": "DSC-FISTA-Iter+WCC",
    "flow": "DSC-Flow",
    "flow+wcc": "DSC-Flow+WCC",
    "flow-iter": "DSC-Flow-Iter",
    "flow-iter+wcc": "DSC-Flow-Iter+WCC",
    # Triangle methods
    "RTRex": "RTRex",
    "RTRex+wcc": "RTRex+WCC",
    # Leiden methods
    "leiden-mod": "Leiden-Mod",
    "leiden-cpm-0.1": "Leiden-CPM(0.1)",
    "leiden-cpm-0.01": "Leiden-CPM(0.01)",
    "leiden-cpm-0.001": "Leiden-CPM(0.001)",
    "leiden-cpm-0.0001": "Leiden-CPM(0.0001)",
    "leiden-mod+wcc": "Leiden-Mod+WCC",
    "leiden-cpm-0.1+wcc": "Leiden-CPM(0.1)+WCC",
    "leiden-cpm-0.01+wcc": "Leiden-CPM(0.01)+WCC",
    "leiden-cpm-0.001+wcc": "Leiden-CPM(0.001)+WCC",
    "leiden-cpm-0.0001+wcc": "Leiden-CPM(0.0001)+WCC",
    # Infomap methods
    "infomap+cc": "Infomap",
    "infomap+wcc": "Infomap+WCC",
    # IKC methods
    "ikc-1": "IKC(1)",
    "ikc-2": "IKC(2)",
    "ikc-5": "IKC(5)",
    "ikc-10": "IKC(10)",
    "ikc-20": "IKC(20)",
    "ikc-1+wcc": "IKC(1)+WCC",
    "ikc-2+wcc": "IKC(2)+WCC",
    "ikc-5+wcc": "IKC(5)+WCC",
    "ikc-10+wcc": "IKC(10)+WCC",
    "ikc-20+wcc": "IKC(20)+WCC",
    # Ensemble methods
    # DSC-Flow-Iter x Leiden-Mod combinations
    # Fully-weighted
    "flow-iter-x-leiden-mod--0-0.0--leiden-cpm-0.01+wcc": "FMC-Constrained-Full",
    "flow-iter-x-leiden-mod--1-0.0--leiden-cpm-0.01+wcc": "FMC-Free-Full",
    # Majority rule (W)
    "flow-iter-x-leiden-mod--0-0.5--leiden-cpm-0.01+wcc": "FMC-Constrained-Majority(W)",
    "flow-iter-x-leiden-mod--1-1--leiden-cpm-0.01+wcc": "FMC-Free-Majority(W)",
    # Majority rule (U)
    "flow-iter-x-leiden-mod--0-0.5-U--leiden-cpm-0.01+wcc": "FMC-Constrained-Majority(U)",
    "flow-iter-x-leiden-mod--1-1-U--leiden-cpm-0.01+wcc": "FMC-Free-Majority(U)",
    # Strict consensus
    "flow-iter-x-leiden-mod--0-1.0-U--leiden-cpm-0.01+wcc": "FMC-Constrained-Strict",
    "flow-iter-x-leiden-mod--1-2-U--leiden-cpm-0.01+wcc": "FMC-Free-Strict",
    # DSC-Flow-Iter x Leiden-Mod x RTRex combinations
    # Fully-weighted
    "flow-iter-x-leiden-mod-x-RTRex--0-0.0--leiden-cpm-0.01+wcc": "FMRC-Constrained-Full",
    "flow-iter-x-leiden-mod-x-RTRex--1-0.0--leiden-cpm-0.01+wcc": "FMRC-Free-Full",
    # Majority rule (W)
    "flow-iter-x-leiden-mod-x-RTRex--0-0.5--leiden-cpm-0.01+wcc": "FMRC-Constrained-Majority(W)",
    "flow-iter-x-leiden-mod-x-RTRex--1-2--leiden-cpm-0.01+wcc": "FMRC-Free-Majority(W)",
    # Majority rule (U)
    "flow-iter-x-leiden-mod-x-RTRex--0-0.5-U--leiden-cpm-0.01+wcc": "FMRC-CVC",
    "flow-iter-x-leiden-mod-x-RTRex--1-2-U--leiden-cpm-0.01+wcc": "FMRC-Free-Majority(U)",
    # Strict consensus
    "flow-iter-x-leiden-mod-x-RTRex--0-1.0-U--leiden-cpm-0.01+wcc": "FMRC-Constrained-Strict",
    "flow-iter-x-leiden-mod-x-RTRex--1-3-U--leiden-cpm-0.01+wcc": "FMRC-Free-Strict",
    # DSC-Flow-Iter x Leiden-Mod x RTRex x IKC(5) combinations
    # Fully-weighted
    "flow-iter-x-leiden-mod-x-RTRex-x-ikc-5--0-0.0--leiden-cpm-0.01+wcc": "FMRKC-Constrained-Full",
    "flow-iter-x-leiden-mod-x-RTRex-x-ikc-5--1-0.0--leiden-cpm-0.01+wcc": "FMRKC-Free-Full",
    # Majority rule (W)
    "flow-iter-x-leiden-mod-x-RTRex-x-ikc-5--0-0.5--leiden-cpm-0.01+wcc": "FMRKC-Constrained-Majority(W)",
    "flow-iter-x-leiden-mod-x-RTRex-x-ikc-5--1-2--leiden-cpm-0.01+wcc": "FMRKC-Free-Majority(W)",
    # Majority rule (U)
    "flow-iter-x-leiden-mod-x-RTRex-x-ikc-5--0-0.5-U--leiden-cpm-0.01+wcc": "FMRKC-CVC",
    "flow-iter-x-leiden-mod-x-RTRex-x-ikc-5--1-2-U--leiden-cpm-0.01+wcc": "FMRKC-FVC",
    # Strict majority rule (U)
    "flow-iter-x-leiden-mod-x-RTRex-x-ikc-5--0-0.6-U--leiden-cpm-0.01+wcc": "FMRKC-Constrained-StrictMajority(U)",
    "flow-iter-x-leiden-mod-x-RTRex-x-ikc-5--1-3-U--leiden-cpm-0.01+wcc": "FMRKC-Free-StrictMajority(U)",
    # Strict majority rule (W)
    "flow-iter-x-leiden-mod-x-RTRex-x-ikc-5--0-0.6--leiden-cpm-0.01+wcc": "FMRKC-Constrained-StrictMajority(W)",
    "flow-iter-x-leiden-mod-x-RTRex-x-ikc-5--1-3--leiden-cpm-0.01+wcc": "FMRKC-Free-StrictMajority(W)",
    # Strict consensus
    "flow-iter-x-leiden-mod-x-RTRex-x-ikc-5--0-1.0-U--leiden-cpm-0.01+wcc": "FMRKC-Constrained-Strict",
    "flow-iter-x-leiden-mod-x-RTRex-x-ikc-5--1-4-U--leiden-cpm-0.01+wcc": "FMRKC-Free-Strict",
    # pamcon
    "leiden-mod-x-leiden-cpm-0.01+wcc--pamcon": "MC-MedCon",
    "flow-iter-x-leiden-mod-x-RTRex-x-ikc-5-x-leiden-cpm-0.01+wcc--pamcon": "FMRKC-MedCon",
    # Ablation
    "leiden-mod-x-RTRex-x-ikc-5--0-0.5-U--leiden-cpm-0.01+wcc": "MRKC-CVC",
    "flow-iter-x-RTRex-x-ikc-5--0-0.5-U--leiden-cpm-0.01+wcc": "FRKC-CVC",
    "flow-iter-x-leiden-mod-x-ikc-5--0-0.5-U--leiden-cpm-0.01+wcc": "FMKC-CVC",
}

# --- Helper Functions ---


def q1(x):
    """Calculate the 25th percentile."""
    return x.quantile(0.25)


def q3(x):
    """Calculate the 75th percentile."""
    return x.quantile(0.75)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process graph clustering logs and generate plots."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help="Path to the input CSV file containing clustering metrics",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Directory to save output plots",
    )
    return parser.parse_args()


def get_common_networks_df(
    df: pd.DataFrame,
    methods: List[str],
    metrics: List[str],
    network_list: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Filters the DataFrame to include only:
    1. The specified methods.
    2. Networks present in `network_list`.
    3. Networks where ALL specified methods have valid (non-NaN) data for ALL metrics.
    """
    # 1. Map keys to labels
    target_labels = []
    for m in methods:
        if m in MAPPING:
            target_labels.append(MAPPING[m])
        else:
            logger.warning(f"Method key '{m}' not found in MAPPING. Skipping.")

    if not target_labels:
        logger.warning("No valid methods found.")
        return pd.DataFrame()

    # 2. Filter by Network List
    if network_list is not None:
        df = df[df["network_id"].isin(network_list)].copy()

    # 3. Filter by Method Labels
    df_plot = df[df["Method"].isin(target_labels)].copy()

    # 4. Strict Intersection Filtering
    valid_networks = []
    for net_id, group in df_plot.groupby("network_id"):
        # Check: Are all methods present?
        if group["Method"].nunique() != len(target_labels):
            continue
        # Check: Are there any NaNs?
        if group[metrics].isna().any().any():
            continue
        valid_networks.append(net_id)

    df_clean = df_plot[df_plot["network_id"].isin(valid_networks)].copy()

    # Set categorical order for consistency
    df_clean["Method"] = pd.Categorical(
        df_clean["Method"], categories=target_labels, ordered=True
    )
    df_clean.sort_values("Method", inplace=True)

    return df_clean, target_labels


def generate_summary_table(
    df: pd.DataFrame,
    methods: List[str],
    metrics: List[str],
    output_dir: Path,
    output_fn: str,
    network_list: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Generates a summary CSV containing stats and 'count_best' for the given methods/metrics.
    Also generates a pairwise t-test report.
    Returns the cleaned DataFrame used for calculations.
    """
    logger.info(f"Generating Summary Table: {output_fn}")

    df_clean, target_labels = get_common_networks_df(df, methods, metrics, network_list)

    if df_clean.empty:
        logger.warning("Cleaned DataFrame is empty. Skipping summary generation.")
        return pd.DataFrame()

    # A. Calculate Standard Stats
    stats_df = df_clean.groupby("Method", observed=True)[metrics].agg(
        ["count", "min", q1, "median", q3, "max", "mean", "std"]
    )
    stats_stacked = stats_df.stack(level=1, future_stack=True)

    # B. Calculate 'count_best'
    methods_idx = stats_stacked.index.get_level_values(0).unique()
    best_counts_df = pd.DataFrame(0, index=methods_idx, columns=metrics)

    for metric in metrics:
        pivoted = df_clean.pivot(index="network_id", columns="Method", values=metric)
        max_vals = pivoted.max(axis=1)
        is_best = pivoted.eq(max_vals, axis=0)
        counts = is_best.sum(axis=0)
        best_counts_df[metric] = counts

    best_counts_df["Stat"] = "count_best"
    best_counts_df = best_counts_df.set_index("Stat", append=True)

    # C. Combine and Save CSV
    final_summary = pd.concat([stats_stacked, best_counts_df]).sort_index()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"summary_{output_fn}.csv"
    final_summary.reset_index().to_csv(out_path, index=False)
    logger.info(f"Saved summary to {out_path}")

    return df_clean


def plot_boxplots(
    df_clean: pd.DataFrame,
    metrics: List[str],
    metric_names: List[str],
    output_dir: Path,
    output_fn: str,
    xlim: Optional[tuple] = None,
    show_vline: bool = False,
    vline_x: float = 0.0,
    vline_kwargs: Optional[dict] = None,
):
    """
    Generates boxplots from a pre-cleaned DataFrame.
    """
    if df_clean.empty:
        logger.warning(f"DataFrame empty for plot {output_fn}. Skipping.")
        return

    logger.info(f"Generating Plot: {output_fn}")

    # Get target labels from the categorical column to ensure order
    target_labels = df_clean["Method"].cat.categories.tolist()

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(metrics),
        figsize=(4 * len(metrics) + 2, len(target_labels) * 0.5 + 2),
        dpi=300,
        constrained_layout=True,
        sharey=True,
    )

    if len(metrics) == 1:
        axes = [axes]

    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx]

        ax.grid(True, which="major", axis="x", linestyle="--", color="lightgray")

        sns.boxplot(
            data=df_clean,
            x=metric,
            y="Method",
            ax=ax,
            order=target_labels,
            color="white",
            showmeans=True,
            orient="h",
            boxprops={"edgecolor": "black", "linewidth": 1.5},
            whiskerprops={"color": "black", "linewidth": 1.5},
            capprops={"color": "black", "linewidth": 1.5},
            medianprops={"color": "red", "linewidth": 1},
            meanprops={
                "marker": "^",
                "markerfacecolor": "green",
                "markeredgecolor": "green",
                "markersize": 5,
            },
            flierprops={
                "marker": "o",
                "markerfacecolor": "black",
                "markeredgecolor": "black",
                "markersize": 3,
            },
        )

        if show_vline:
            kwargs = vline_kwargs or {
                "color": "red",
                "linestyle": "--",
                "linewidth": 1,
                "alpha": 0.7,
            }
            ax.axvline(x=vline_x, **kwargs)

        if xlim:
            ax.set_xlim(*xlim)

        ax.set_xlabel(metric_name, fontsize=14)

        if idx == 0:
            ax.set_ylabel("Method", fontsize=14)
        else:
            ax.set_ylabel("")

        if "user_time" in metric or "max_mem_usage" in metric:
            ax.set_xscale("log")

        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=12)

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / output_fn, bbox_inches="tight")
    logger.info(f"Saved plot to {output_dir / output_fn}")
    plt.close(fig)


# --- Main Execution ---

if __name__ == "__main__":
    args = parse_args()
    csv_path = Path(args.input_csv)
    output_dir = Path(args.output_dir)

    logger.info(f"Loading existing data from {csv_path}")
    df = pd.read_csv(csv_path)

    # Plotting or generating tables
    methods = [
        "fista-int",
        "flow",
        "fista-frac-iter",
        "fista-int-iter",
        "flow-iter",
    ]
    file = "journal_exp1_clustering"

    df_subset = generate_summary_table(
        df,
        methods,
        ["ami", "ari", "nmi"],
        output_dir / "overall",
        file,
    )

    plot_boxplots(
        df_subset,
        ["ami", "ari", "nmi"],
        ["AMI", "ARI", "NMI"],
        output_dir / "overall",
        f"{file}.pdf",
        xlim=(0.0, 1.0),
    )

    methods = [
        "flow-iter",
        "RTRex",
        "leiden-cpm-0.1",
        "leiden-cpm-0.01",
        "leiden-cpm-0.001",
        "leiden-cpm-0.0001",
        "leiden-mod",
        "infomap+cc",
        "ikc-1",
        "ikc-2",
        "ikc-5",
        "ikc-10",
        "ikc-20",
    ]
    file = "journal_exp2_all_clustering"

    df_subset = generate_summary_table(
        df,
        methods,
        ["ami", "ari", "nmi"],
        output_dir / "overall",
        file,
    )

    plot_boxplots(
        df_subset,
        ["ami", "ari", "nmi"],
        ["AMI", "ARI", "NMI"],
        output_dir / "overall",
        f"{file}.pdf",
        xlim=(0.0, 1.0),
    )

    methods = [
        "flow-iter",
        "RTRex",
        "ikc-1",
        "leiden-cpm-0.01",
        "leiden-mod",
        "infomap+cc",
    ]
    file = "journal_exp2_subset_clustering"

    df_subset = generate_summary_table(
        df,
        methods,
        ["ami", "ari", "nmi", "node_coverage"],
        output_dir / "overall",
        file,
    )

    plot_boxplots(
        df_subset,
        ["ami", "ari", "nmi"],
        ["AMI", "ARI", "NMI"],
        output_dir / "overall",
        f"{file}.pdf",
        xlim=(0.0, 1.0),
    )

    # Train
    train_network_ids = None
    net_file = Path("data/networks_train.txt")
    if net_file.exists():
        with open(net_file, "r") as f:
            train_network_ids = [line.strip() for line in f.readlines() if line.strip()]
    else:
        logger.warning(f"{net_file} not found. Skipping network filtering.")

    methods = [
        "flow-iter",
        "flow-iter+wcc",
        "RTRex",
        "RTRex+wcc",
        "leiden-cpm-0.1",
        "leiden-cpm-0.1+wcc",
        "leiden-cpm-0.01",
        "leiden-cpm-0.01+wcc",
        "leiden-cpm-0.001",
        "leiden-cpm-0.001+wcc",
        "leiden-cpm-0.0001",
        "leiden-cpm-0.0001+wcc",
        "leiden-mod",
        "leiden-mod+wcc",
        "infomap+cc",
        "infomap+wcc",
        "ikc-1",
        "ikc-1+wcc",
        "ikc-2",
        "ikc-2+wcc",
        "ikc-5",
        "ikc-5+wcc",
        "ikc-10",
        "ikc-10+wcc",
        "ikc-20",
        "ikc-20+wcc",
    ]
    file = "journal_exp3_train_choose-best_clustering"

    df_subset = generate_summary_table(
        df,
        methods,
        ["ami", "ari", "nmi"],
        output_dir / "train",
        file,
        train_network_ids,
    )

    plot_boxplots(
        df_subset,
        ["ami", "ari", "nmi"],
        ["AMI", "ARI", "NMI"],
        output_dir / "train",
        f"{file}.pdf",
        xlim=(0.0, 1.0),
    )

    methods = [
        "flow-iter",
        "flow-iter+wcc",
        "RTRex",
        "RTRex+wcc",
        "leiden-cpm-0.01",
        "leiden-cpm-0.01+wcc",
        "leiden-mod",
        "leiden-mod+wcc",
        "infomap+cc",
        "infomap+wcc",
        "ikc-1",
        "ikc-1+wcc",
    ]
    file = "journal_exp3_train_choose-best_subset_clustering"

    df_subset = generate_summary_table(
        df,
        methods,
        ["ami", "ari", "nmi"],
        output_dir / "train",
        file,
        train_network_ids,
    )

    plot_boxplots(
        df_subset,
        ["ami", "ari", "nmi"],
        ["AMI", "ARI", "NMI"],
        output_dir / "train",
        f"{file}.pdf",
        xlim=(0.0, 1.0),
    )

    methods = [
        "flow-iter",
        "RTRex",
        "leiden-cpm-0.1",
        "leiden-cpm-0.01",
        "leiden-cpm-0.001",
        "leiden-cpm-0.0001",
        "leiden-mod",
        "infomap+cc",
        "ikc-1",
        "ikc-2",
        "ikc-5",
        "ikc-10",
        "ikc-20",
    ]
    file = "journal_exp3_train_choose-ens_confusion"

    df_subset = generate_summary_table(
        df,
        methods,
        ["f1_score", "precision", "recall", "node_coverage"],
        output_dir / "train",
        file,
        train_network_ids,
    )

    methods = [
        # 2 clusterings
        # Fully-weighted
        "flow-iter-x-leiden-mod--0-0.0--leiden-cpm-0.01+wcc",
        "flow-iter-x-leiden-mod--1-0.0--leiden-cpm-0.01+wcc",
        # Majority rule (W)
        "flow-iter-x-leiden-mod--0-0.5--leiden-cpm-0.01+wcc",
        "flow-iter-x-leiden-mod--1-1--leiden-cpm-0.01+wcc",
        # Majority rule (U)
        "flow-iter-x-leiden-mod--0-0.5-U--leiden-cpm-0.01+wcc",
        "flow-iter-x-leiden-mod--1-1-U--leiden-cpm-0.01+wcc",
        # Strict consensus
        "flow-iter-x-leiden-mod--0-1.0-U--leiden-cpm-0.01+wcc",
        "flow-iter-x-leiden-mod--1-2-U--leiden-cpm-0.01+wcc",
        # 3 clusterings
        # Fully-weighted
        "flow-iter-x-leiden-mod-x-RTRex--0-0.0--leiden-cpm-0.01+wcc",
        "flow-iter-x-leiden-mod-x-RTRex--1-0.0--leiden-cpm-0.01+wcc",
        # Majority rule (W)
        "flow-iter-x-leiden-mod-x-RTRex--0-0.5--leiden-cpm-0.01+wcc",
        "flow-iter-x-leiden-mod-x-RTRex--1-2--leiden-cpm-0.01+wcc",
        # Majority rule (U)
        "flow-iter-x-leiden-mod-x-RTRex--0-0.5-U--leiden-cpm-0.01+wcc",
        "flow-iter-x-leiden-mod-x-RTRex--1-2-U--leiden-cpm-0.01+wcc",
        # Strict consensus
        "flow-iter-x-leiden-mod-x-RTRex--0-1.0-U--leiden-cpm-0.01+wcc",
        "flow-iter-x-leiden-mod-x-RTRex--1-3-U--leiden-cpm-0.01+wcc",
        # 4 clusterings
        # Fully-weighted
        "flow-iter-x-leiden-mod-x-RTRex-x-ikc-5--0-0.0--leiden-cpm-0.01+wcc",
        "flow-iter-x-leiden-mod-x-RTRex-x-ikc-5--1-0.0--leiden-cpm-0.01+wcc",
        # Majority rule (W)
        "flow-iter-x-leiden-mod-x-RTRex-x-ikc-5--0-0.5--leiden-cpm-0.01+wcc",
        "flow-iter-x-leiden-mod-x-RTRex-x-ikc-5--1-2--leiden-cpm-0.01+wcc",
        # Majority rule (U)
        "flow-iter-x-leiden-mod-x-RTRex-x-ikc-5--0-0.5-U--leiden-cpm-0.01+wcc",
        "flow-iter-x-leiden-mod-x-RTRex-x-ikc-5--1-2-U--leiden-cpm-0.01+wcc",
        # Strict majority rule (U)
        "flow-iter-x-leiden-mod-x-RTRex-x-ikc-5--0-0.6-U--leiden-cpm-0.01+wcc",
        "flow-iter-x-leiden-mod-x-RTRex-x-ikc-5--1-3-U--leiden-cpm-0.01+wcc",
        # Strict majority rule (W)
        "flow-iter-x-leiden-mod-x-RTRex-x-ikc-5--0-0.6--leiden-cpm-0.01+wcc",
        "flow-iter-x-leiden-mod-x-RTRex-x-ikc-5--1-3--leiden-cpm-0.01+wcc",
        # Strict consensus
        "flow-iter-x-leiden-mod-x-RTRex-x-ikc-5--0-1.0-U--leiden-cpm-0.01+wcc",
        "flow-iter-x-leiden-mod-x-RTRex-x-ikc-5--1-4-U--leiden-cpm-0.01+wcc",
    ]

    generate_summary_table(
        df,
        methods,
        ["ami", "ari", "nmi"],
        output_dir / "train",
        "journal_exp3_test_all-consensus_clustering",
        train_network_ids,
    )

    generate_summary_table(
        df,
        methods,
        ["f1_score", "precision", "recall"],
        output_dir / "train",
        "journal_exp3_test_all-consensus_confusion",
        train_network_ids,
    )

    # Test
    test_network_ids = None
    net_file = Path("data/networks_test.txt")
    if net_file.exists():
        with open(net_file, "r") as f:
            test_network_ids = [line.strip() for line in f.readlines() if line.strip()]
    else:
        logger.warning(f"{net_file} not found. Skipping network filtering.")

    methods = [
        "leiden-cpm-0.01+wcc",
        "flow-iter-x-leiden-mod-x-RTRex-x-ikc-5--0-0.5-U--leiden-cpm-0.01+wcc",
    ]
    file = "journal_exp3_test_baseline_clustering"

    df_subset = generate_summary_table(
        df,
        methods,
        ["ami", "ari", "nmi"],
        output_dir / "test",
        file,
        test_network_ids,
    )

    plot_boxplots(
        df_subset,
        ["ami", "ari", "nmi"],
        ["AMI", "ARI", "NMI"],
        output_dir / "test",
        f"{file}.pdf",
        xlim=(0.0, 1.0),
    )

    # --- Configuration 1: Consensus Clustering ---
    methods_consensus = [
        "flow-iter-x-leiden-mod-x-RTRex-x-ikc-5--0-0.5-U--leiden-cpm-0.01+wcc",
        "flow-iter-x-leiden-mod-x-RTRex-x-ikc-5--1-2-U--leiden-cpm-0.01+wcc",
        "flow-iter-x-leiden-mod-x-RTRex-x-ikc-5-x-leiden-cpm-0.01+wcc--pamcon",
    ]
    metrics_consensus = ["ami", "ari", "nmi"]
    names_consensus = ["AMI", "ARI", "NMI"]
    file_consensus = "journal_exp3_test_consensus_clustering"

    # 1. Generate Table & Get Clean Data
    df_cons_clean = generate_summary_table(
        df,
        methods_consensus,
        metrics_consensus,
        output_dir / "test",
        file_consensus,
        test_network_ids,
    )

    # 2. Plot using Clean Data
    plot_boxplots(
        df_cons_clean,
        metrics_consensus,
        names_consensus,
        output_dir / "test",
        f"{file_consensus}.pdf",
        xlim=(0.0, 1.0),
    )

    # --- Configuration 2: Confusion Matrix ---
    methods_confusion = [
        "leiden-mod",
        "RTRex",
        "flow-iter-x-leiden-mod-x-RTRex-x-ikc-5--0-0.5-U--leiden-cpm-0.01+wcc",
        "flow-iter-x-leiden-mod-x-RTRex-x-ikc-5-x-leiden-cpm-0.01+wcc--pamcon",
    ]
    metrics_confusion = ["f1_score", "precision", "recall"]
    names_confusion = ["F1", "Precision", "Recall"]
    file_confusion = "journal_exp3_test_consensus_confusion"

    # 1. Generate Table & Get Clean Data
    df_conf_clean = generate_summary_table(
        df,
        methods_confusion,
        metrics_confusion,
        output_dir / "test",
        file_confusion,
        test_network_ids,
    )

    # 2. Plot using Clean Data
    plot_boxplots(
        df_conf_clean,
        metrics_confusion,
        names_confusion,
        output_dir / "test",
        f"{file_confusion}.pdf",
        xlim=(0.0, 1.0),
    )

    # --- Configuration 3 ---
    methods = [
        "flow-iter",
        "leiden-mod",
        "RTRex",
        "ikc-5",
        "leiden-cpm-0.01+wcc",
        "flow-iter-x-leiden-mod-x-RTRex-x-ikc-5--0-0.5-U--leiden-cpm-0.01+wcc",
        "flow-iter-x-leiden-mod-x-RTRex-x-ikc-5--1-2-U--leiden-cpm-0.01+wcc",
        "flow-iter-x-leiden-mod-x-RTRex-x-ikc-5-x-leiden-cpm-0.01+wcc--pamcon",
    ]

    generate_summary_table(
        df,
        methods,
        ["ami", "ari", "nmi", "node_coverage"],
        output_dir / "test",
        "journal_exp3_test_all_clustering",
        test_network_ids,
    )

    generate_summary_table(
        df,
        methods,
        ["f1_score", "precision", "recall", "node_coverage"],
        output_dir / "test",
        "journal_exp3_test_all_confusion",
        test_network_ids,
    )

    # ---

    methods = [
        "flow-iter",
        "leiden-mod",
        "RTRex",
        "ikc-5",
        "leiden-cpm-0.01+wcc",
    ]

    generate_summary_table(
        df,
        methods,
        ["node_coverage"],
        output_dir / "test",
        "journal_exp3_test_constituent_coverage",
        test_network_ids,
    )

    # ---

    methods_confusion = [
        "flow-iter-x-leiden-mod-x-RTRex-x-ikc-5--0-0.5-U--leiden-cpm-0.01+wcc",
        "leiden-mod-x-leiden-cpm-0.01+wcc--pamcon",
        "flow-iter-x-leiden-mod-x-RTRex-x-ikc-5-x-leiden-cpm-0.01+wcc--pamcon",
    ]
    metrics_confusion = ["f1_score", "precision", "recall"]
    names_confusion = ["F1", "Precision", "Recall"]
    file_confusion = "journal_exp3_test_medcon_confusion"

    # 1. Generate Table & Get Clean Data
    df_conf_clean = generate_summary_table(
        df,
        methods_confusion,
        metrics_confusion,
        output_dir / "test",
        file_confusion,
        test_network_ids,
    )

    # 2. Plot using Clean Data
    plot_boxplots(
        df_conf_clean,
        metrics_confusion,
        names_confusion,
        output_dir / "test",
        f"{file_confusion}.pdf",
        xlim=(0.0, 1.0),
    )

    # --

    methods = [
        "flow-iter-x-leiden-mod-x-RTRex-x-ikc-5--0-0.5-U--leiden-cpm-0.01+wcc",
        "leiden-mod-x-RTRex-x-ikc-5--0-0.5-U--leiden-cpm-0.01+wcc",
        "flow-iter-x-leiden-mod-x-RTRex-x-ikc-5-x-leiden-cpm-0.01+wcc--pamcon",
    ]

    df_subset = generate_summary_table(
        df,
        methods,
        ["ami", "ari", "nmi"],
        output_dir / "test",
        "journal_exp3_test_noF_clustering",
        test_network_ids,
    )

    plot_boxplots(
        df_subset,
        ["ami", "ari", "nmi"],
        ["AMI", "ARI", "NMI"],
        output_dir / "test",
        f"journal_exp3_test_noF_clustering.pdf",
        xlim=(0.0, 1.0),
    )

    df_subset = generate_summary_table(
        df,
        methods,
        ["f1_score", "precision", "recall"],
        output_dir / "test",
        "journal_exp3_test_noF_confusion",
        test_network_ids,
    )

    plot_boxplots(
        df_subset,
        ["f1_score", "precision", "recall"],
        ["F1", "Precision", "Recall"],
        output_dir / "test",
        f"journal_exp3_test_noF_confusion.pdf",
        xlim=(0.0, 1.0),
    )

    # ---

    methods = [
        "flow-iter-x-leiden-mod-x-RTRex-x-ikc-5--0-0.5-U--leiden-cpm-0.01+wcc",
        "leiden-mod-x-RTRex-x-ikc-5--0-0.5-U--leiden-cpm-0.01+wcc",
        "flow-iter-x-RTRex-x-ikc-5--0-0.5-U--leiden-cpm-0.01+wcc",
        "flow-iter-x-leiden-mod-x-ikc-5--0-0.5-U--leiden-cpm-0.01+wcc",
        "flow-iter-x-leiden-mod-x-RTRex--0-0.5-U--leiden-cpm-0.01+wcc",
    ]

    df_subset = generate_summary_table(
        df,
        methods,
        ["ami", "ari", "nmi"],
        output_dir / "test",
        "journal_exp3_test_ablation_clustering",
        test_network_ids,
    )

    plot_boxplots(
        df_subset,
        ["ami", "ari", "nmi"],
        ["AMI", "ARI", "NMI"],
        output_dir / "test",
        f"journal_exp3_test_ablation_clustering.pdf",
        xlim=(0.0, 1.0),
    )

    df_subset = generate_summary_table(
        df,
        methods,
        ["f1_score", "precision", "recall"],
        output_dir / "test",
        "journal_exp3_test_ablation_confusion",
        test_network_ids,
    )

    plot_boxplots(
        df_subset,
        ["f1_score", "precision", "recall"],
        ["F1", "Precision", "Recall"],
        output_dir / "test",
        f"journal_exp3_test_ablation_confusion.pdf",
        xlim=(0.0, 1.0),
    )

    logger.info("Processing complete.")

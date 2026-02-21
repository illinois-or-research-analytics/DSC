import argparse
import json
import logging
import sys
import textwrap
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

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
    # "flow-iter+cm": "DSC-Flow-Iter+CM",
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
    # "leiden-mod+cm": "Leiden-Mod+CM",
    # "leiden-cpm-0.1+cm": "Leiden-CPM(0.1)+CM",
    # "leiden-cpm-0.01+cm": "Leiden-CPM(0.01)+CM",
    # "leiden-cpm-0.001+cm": "Leiden-CPM(0.001)+CM",
    # "leiden-cpm-0.0001+cm": "Leiden-CPM(0.0001)+CM",
    # Infomap methods
    "infomap+cc": "Infomap",
    "infomap+wcc": "Infomap+WCC",
    # "infomap+cm": "Infomap+CM",
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
    # "ikc-1+cm": "IKC(1)+CM",
    # "ikc-2+cm": "IKC(2)+CM",
    # "ikc-5+cm": "IKC(5)+CM",
    # "ikc-10+cm": "IKC(10)+CM",
    # "ikc-20+cm": "IKC(20)+CM",
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
    # MedCon
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
        "--log-root",
        type=str,
        default="data/dsc",
        help="Root directory for the log data",
    )
    parser.add_argument(
        "--acc-root",
        type=str,
        default="data/dsc/acc",
        help="Root directory for the accuracy data",
    )
    parser.add_argument(
        "--stats-root",
        type=str,
        default="data/dsc/stats",
        help="Root directory for the stats data",
    )
    parser.add_argument(
        "--network-list",
        type=str,
        default="data/networks_val.txt",
        help="File containing network IDs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Directory to save output plots",
    )
    parser.add_argument(
        "--generator",
        type=str,
        default="ec-sbm",
        help="Synthesis method",
    )
    parser.add_argument(
        "--gt-clustering",
        type=str,
        default="sbm_wcc",
        help="Synthesis empirical clustering",
    )
    parser.add_argument(
        "--gt-id",
        type=str,
        default="0",
        help="Synthesis ID for empirical clustering",
    )
    parser.add_argument(
        "--is-load-existing",
        action="store_true",
        default=False,
        help="Load existing CSV instead of reprocessing",
    )
    parser.add_argument(
        "--n-procs",
        type=int,
        default=16,
        help="Number of processes to use",
    )
    return parser.parse_args()


def _get_log_stats(log_path: Path) -> Dict[str, Optional[float]]:
    """Reads user time and memory usage from a log file."""
    user_time, mem_usage = None, None
    if not log_path.exists():
        return {"user_time": None, "max_mem_usage": None}

    try:
        with open(log_path, "r") as f:
            for line in f:
                if "User time (seconds):" in line:
                    try:
                        user_time = float(line.split(":")[-1].strip())
                    except ValueError:
                        pass
                elif "Maximum resident set size (kbytes):" in line:
                    try:
                        mem_usage = float(line.split(":")[-1].strip())
                    except ValueError:
                        pass
    except Exception as e:
        logger.warning(f"Failed to read log file {log_path}: {e}")

    return {"user_time": user_time, "max_mem_usage": mem_usage}


def _process_task(args: tuple) -> Dict[str, Any]:
    """
    Worker function to process a single network_id and weight combination.
    """
    (
        network_id,
        weight,
        log_root,
        acc_root,
        stats_root,
        generator,
        gt_clustering,
        gt_id,
        metrics_list,
    ) = args

    path_components = [
        generator,
        gt_clustering,
        network_id,
        gt_id,
        weight,
    ]

    stats_path = stats_root.joinpath(*path_components)
    acc_path = acc_root.joinpath(*path_components)

    row_data = {"network_id": network_id, "weight": weight}

    # --- 1. Process Cluster Connectivity (Stats) ---
    try:
        stats_file = stats_path / "stats.json"
        if stats_file.exists():
            with open(stats_file, "r") as f:
                connectivity = json.load(f)

            n_clusters = connectivity.get("n_clusters", 0)
            n_disconnected = connectivity.get("n_disconnects", 0)

            row_data.update(
                {
                    "n_clusters": n_clusters,
                    "n_singleton": connectivity.get("n_onodes"),
                    "n_disconnected": n_disconnected,
                    "n_wellconnected": connectivity.get("n_wellconnected_clusters"),
                }
            )

            if n_clusters > 0:
                ratio_disconnected = n_disconnected / n_clusters
                row_data["ratio_disconnected"] = ratio_disconnected
                row_data["ratio_wellconnected"] = (
                    row_data.get("n_wellconnected", 0) / n_clusters
                )
                row_data["n_connected"] = n_clusters - n_disconnected
                row_data["ratio_connected"] = 1.0 - ratio_disconnected
            else:
                row_data.update(
                    {
                        "ratio_disconnected": 0.0,
                        "ratio_wellconnected": 0.0,
                        "n_connected": 0,
                        "ratio_connected": 0.0,
                    }
                )
        else:
            for key in [
                "n_clusters",
                "n_singleton",
                "n_disconnected",
                "n_wellconnected",
                "ratio_disconnected",
                "ratio_wellconnected",
                "n_connected",
                "ratio_connected",
            ]:
                row_data[key] = None

    except Exception as e:
        logger.error(f"Error reading stats for {network_id}/{weight}: {e}")
        row_data["n_clusters"] = None

    # --- 2. Read Accuracy Metrics ---
    for metric in metrics_list:
        metric_file = acc_path / f"result.{metric}"
        try:
            if metric_file.exists():
                with open(metric_file, "r") as f:
                    val = float(f.read().strip())
                    row_data[metric] = val
                    if metric in ["fpr", "fnr"]:
                        row_data[f"comp_{metric}"] = 1.0 - val
            else:
                row_data[metric] = None
        except ValueError:
            row_data[metric] = None
        except Exception:
            row_data[metric] = None

    # --- 3. Read Log Files (Time/Mem) ---
    if weight.endswith(("+cc", "+wcc")):
        suffix_len = 3 if weight.endswith("+cc") else 4
        base_weight = weight[:-suffix_len]
        processing_weight = weight
    else:
        base_weight = weight
        processing_weight = None

    log_base_comps = [
        generator,
        gt_clustering,
        network_id,
        gt_id,
    ]

    base_log_path = log_root.joinpath(*log_base_comps, base_weight, "error.log")
    base_stats = _get_log_stats(base_log_path)

    if processing_weight:
        proc_log_path = log_root.joinpath(
            *log_base_comps, processing_weight, "error.log"
        )
        processing_stats = _get_log_stats(proc_log_path)
    else:
        processing_stats = {"user_time": 0.0, "max_mem_usage": 0.0}

    row_data["base_user_time"] = base_stats["user_time"]
    row_data["base_max_mem_usage"] = base_stats["max_mem_usage"]
    row_data["processing_user_time"] = processing_stats["user_time"]
    row_data["processing_max_mem_usage"] = processing_stats["max_mem_usage"]

    if (
        base_stats["user_time"] is not None
        and processing_stats["user_time"] is not None
    ):
        row_data["total_user_time"] = (
            base_stats["user_time"] + processing_stats["user_time"]
        )
    else:
        row_data["total_user_time"] = None

    if (
        base_stats["max_mem_usage"] is not None
        and processing_stats["max_mem_usage"] is not None
    ):
        row_data["total_max_mem_usage"] = max(
            base_stats["max_mem_usage"], processing_stats["max_mem_usage"]
        )
    else:
        row_data["total_max_mem_usage"] = None

    return row_data


def collect_dataframe(
    network_ids: List[str],
    weights: List[str],
    log_root: Path,
    acc_root: Path,
    stats_root: Path,
    generator: str,
    gt_clustering: str,
    gt_id: str,
    max_workers: int = 8,
) -> pd.DataFrame:
    metrics_to_read = [
        "agri",
        "ami",
        "ari",
        "nmi",
        "node_coverage",
        "f1_score",
        "fnr",
        "fpr",
        "precision",
        "recall",
    ]

    tasks = []
    for network_id in network_ids:
        for weight in weights:
            tasks.append(
                (
                    network_id,
                    weight,
                    log_root,
                    acc_root,
                    stats_root,
                    generator,
                    gt_clustering,
                    gt_id,
                    metrics_to_read,
                )
            )

    logger.info(
        f"Starting data collection for {len(tasks)} tasks using {max_workers} workers."
    )

    df_data = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(_process_task, task): task for task in tasks}

        for future in tqdm(
            as_completed(future_to_task),
            total=len(tasks),
            desc="Processing Metrics",
            unit="file",
        ):
            try:
                result = future.result()
                if result:
                    df_data.append(result)
            except Exception as e:
                task_info = future_to_task[future]
                logger.error(
                    f"Task failed for Net: {task_info[0]}, Weight: {task_info[1]}. Error: {e}"
                )

    if not df_data:
        logger.error("No data collected. Please check paths and inputs.")
        return pd.DataFrame()

    logger.info(f"Collected {len(df_data)} records. Formatting DataFrame...")

    df = pd.DataFrame(df_data)
    df["weight"] = pd.Categorical(df["weight"], categories=weights, ordered=True)
    df = df.sort_values(by=["network_id", "weight"]).reset_index(drop=True)

    df = df.rename(columns={"weight": "Method"})
    if MAPPING:
        df["Method"] = df["Method"].map(MAPPING)

    return df


def print_completion_summary(df: pd.DataFrame):
    """
    Counts and prints the number of networks that have successfully finished
    (i.e., contain valid metrics) for each Method, and lists the missing ones.
    """
    logger.info("Computing completion statistics...")

    # Identify the full set of networks present in the DataFrame
    # Note: df is expected to contain rows for all networks/methods even if metrics are NaN
    all_networks = set(df["network_id"].unique())
    total_networks = len(all_networks)

    # We consider a task "finished" if at least one key metric is present.
    # 'ami' is standard; if it's NaN, the run likely failed or didn't produce output.
    finished_df = df.dropna(subset=["ami"])

    # Calculate finished sets per method
    method_finished = (
        finished_df.groupby("Method", observed=True)["network_id"].apply(set).to_dict()
    )

    # We want to iterate over all methods present in the Mapping or DataFrame
    all_methods = sorted(df["Method"].unique())

    # Build a list of (method, count, missing_list)
    summary_data = []
    for method in all_methods:
        finished_set = method_finished.get(method, set())
        missing_set = all_networks - finished_set
        summary_data.append((method, len(finished_set), sorted(list(missing_set))))

    # Sort by count (descending), then method name
    summary_data.sort(key=lambda x: (x[1], x[0]), reverse=True)

    print("\n" + "=" * 80)
    print(f"COMPLETION SUMMARY (Total Expected: {total_networks})")
    print("=" * 80)
    print(f"{'Method':<40} | {'Count':<10}")
    print("-" * 80)

    for method, count, missing in summary_data:
        print(f"{method:<40} | {count:<10}")
        if missing:
            prefix = "  Missing: "
            wrapper = textwrap.TextWrapper(
                initial_indent=prefix,
                subsequent_indent=" " * len(prefix),
                width=80,
            )
            print(wrapper.fill(", ".join(missing)))

    print("=" * 80 + "\n")


def plot_boxplots(
    df: pd.DataFrame,
    weights: List[str],
    metrics: List[str],
    metric_names: List[str],
    ylim: Optional[tuple] = None,
    show_hline: bool = False,
    hline_y: float = 0.0,
    hline_kwargs: Optional[dict] = None,
    output_dir: Optional[Path] = None,
    output_fn: Optional[str] = None,
    network_list: Optional[List[str]] = None,
):
    logger.info(f"Generating boxplot: {output_fn} for metrics: {metrics}")

    # --- Filtering by Network List (if provided) ---
    if network_list is not None:
        logger.info(
            f"Filtering plot to specific network list ({len(network_list)} networks)."
        )
        df_plot = df[df["network_id"].isin(network_list)].copy()
        if df_plot.empty:
            logger.warning("Network filtering resulted in empty DataFrame. Skipping.")
            return
    else:
        df_plot = df

    mapped_methods = [MAPPING[w] for w in weights if w in MAPPING]
    if not mapped_methods:
        logger.warning("No valid methods found in mapping. Exiting plot function.")
        return

    df_relevant = df_plot[df_plot["Method"].isin(mapped_methods)].copy()

    # --- Missing Data Report ---
    print("\n" + "=" * 50)
    print(f"MISSING DATA REPORT (Diff Plot: {output_fn})")
    print("=" * 50)
    networks_with_issues = False

    unique_networks = df_relevant["network_id"].unique()
    for net_id in unique_networks:
        net_df = df_relevant[df_relevant["network_id"] == net_id]

        missing_methods = []
        valid_count = 0

        # Check all methods including reference
        for method in mapped_methods:
            method_row = net_df[net_df["Method"] == method]
            is_method_valid = True

            if method_row.empty:
                missing_methods.append(f"{method} (No Row)")
                is_method_valid = False
            else:
                for metric in metrics:
                    if pd.isna(method_row.iloc[0].get(metric)):
                        missing_methods.append(f"{method} (NaN {metric})")
                        is_method_valid = False
                        break

            if is_method_valid:
                valid_count += 1

        # --- Reporting Logic ---
        if valid_count == 0:
            networks_with_issues = True
            print(f"Network: {net_id} -- [NO METHODS FINISHED]")
        elif missing_methods:
            networks_with_issues = True
            print(f"Network: {net_id}")
            for m in missing_methods:
                print(f"  - Missing: {m}")

    if not networks_with_issues:
        print("  All networks have complete data.")
    print("=" * 50 + "\n")

    # --- Filtering and Plotting ---
    df_clean = df_relevant.dropna(subset=metrics)
    # Filter for full completion (standard boxplot logic)
    df_clean = df_clean.groupby("network_id").filter(
        lambda x: len(x) == len(mapped_methods)
    )

    if df_clean.empty:
        logger.warning(
            f"DataFrame empty after filtering for plot {output_fn}. Skipping."
        )
        return

    # if output_dir and output_fn:
    #     summary = (
    #         df_clean.groupby("Method", observed=False)[metrics]
    #         .agg(["count", "min", q1, "median", q3, "max", "mean", "std"])
    #         .stack(future_stack=True)
    #         .reset_index()
    #     )
    #     summary.to_csv(output_dir / f"summary_{output_fn}.csv", index=False)

    fig, axes = plt.subplots(
        nrows=len(metrics),
        ncols=1,
        figsize=(
            len(mapped_methods) * 2 if len(metrics) <= 1 else 12,
            len(metrics) * 5 if len(metrics) <= 1 else len(metrics) * 3,
        ),
        dpi=300,
        tight_layout=True,
        sharex=True,
    )

    if len(metrics) == 1:
        axes = [axes]

    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx]
        ax.grid(True, which="major", axis="y", linestyle="--", color="lightgray")

        sns.boxplot(
            data=df_clean,
            x="Method",
            y=metric,
            ax=ax,
            order=mapped_methods,
            color="white",
            showmeans=True,
            boxprops={"edgecolor": "black", "linewidth": 1.5},
            whiskerprops={"color": "black", "linewidth": 1.5},
            capprops={"color": "black", "linewidth": 1.5},
            medianprops={"color": "red", "linewidth": 1},
            flierprops={
                "marker": "o",
                "markerfacecolor": "black",
                "markeredgecolor": "black",
                "markersize": 3,
            },
        )

        if show_hline:
            kwargs = hline_kwargs or {
                "color": "red",
                "linestyle": "--",
                "linewidth": 1,
                "alpha": 0.7,
            }
            ax.axhline(y=hline_y, **kwargs)

        if ylim:
            ax.set_ylim(*ylim)

        ax.set_ylabel(metric_name, fontsize=14)
        ax.set_xlabel("Method" if idx == len(metrics) - 1 else "", fontsize=14)

        if "user_time" in metric or "max_mem_usage" in metric:
            ax.set_yscale("log")

        plt.setp(
            ax.get_xticklabels(), rotation=90 if len(weights) > 5 else 0, fontsize=14
        )

    if output_dir and output_fn:
        plt.savefig(output_dir / output_fn, bbox_inches="tight")
        logger.info(f"Saved plot to {output_dir / output_fn}")
        plt.close(fig)


def plot_boxplots_diff(
    df: pd.DataFrame,
    weights: List[str],
    ref_weight: str,
    metrics: List[str],
    metric_names: List[str],
    ylim: Optional[tuple] = None,
    show_hline: bool = False,
    hline_y: float = 0.0,
    hline_kwargs: Optional[dict] = None,
    output_dir: Optional[Path] = None,
    output_fn: Optional[str] = None,
    network_list: Optional[List[str]] = None,
):
    logger.info(f"Generating diff plot: {output_fn} relative to {ref_weight}")

    # --- Filtering by Network List (if provided) ---
    if network_list is not None:
        logger.info(
            f"Filtering plot to specific network list ({len(network_list)} networks)."
        )
        df_plot = df[df["network_id"].isin(network_list)].copy()
        if df_plot.empty:
            logger.warning("Network filtering resulted in empty DataFrame. Skipping.")
            return
    else:
        df_plot = df

    mapped_methods = [MAPPING[w] for w in weights if w in MAPPING]
    if ref_weight not in MAPPING:
        logger.error(f"Reference weight {ref_weight} not found in mapping.")
        return

    df_relevant = df_plot[df_plot["Method"].isin(mapped_methods)].copy()

    # --- Missing Data Report ---
    print("\n" + "=" * 50)
    print(f"MISSING DATA REPORT (Diff Plot: {output_fn})")
    print("=" * 50)
    networks_with_issues = False

    unique_networks = df_relevant["network_id"].unique()
    for net_id in unique_networks:
        net_df = df_relevant[df_relevant["network_id"] == net_id]

        missing_methods = []
        valid_count = 0

        # Check all methods including reference
        for method in mapped_methods:
            method_row = net_df[net_df["Method"] == method]
            is_method_valid = True

            if method_row.empty:
                missing_methods.append(f"{method} (No Row)")
                is_method_valid = False
            else:
                for metric in metrics:
                    if pd.isna(method_row.iloc[0].get(metric)):
                        missing_methods.append(f"{method} (NaN {metric})")
                        is_method_valid = False
                        break

            if is_method_valid:
                valid_count += 1

        # --- Reporting Logic ---
        if valid_count == 0:
            networks_with_issues = True
            print(f"Network: {net_id} -- [NO METHODS FINISHED]")
        elif missing_methods:
            networks_with_issues = True
            print(f"Network: {net_id}")
            for m in missing_methods:
                print(f"  - Missing: {m}")

    if not networks_with_issues:
        print("  All networks have complete data.")
    print("=" * 50 + "\n")

    # --- Filtering and Plotting ---
    df_clean = df_relevant.dropna(subset=metrics)
    df_clean = df_clean.groupby("network_id").filter(
        lambda x: len(x) == len(mapped_methods)
    )

    if df_clean.empty:
        logger.warning(
            f"DataFrame empty after filtering for diff plot {output_fn}. Skipping."
        )
        return

    ref_method = MAPPING[ref_weight]
    diff_rows = []

    for network_id, grp in df_clean.groupby("network_id"):
        ref_row = grp[grp["Method"] == ref_method]
        if ref_row.empty:
            continue
        ref_vals = ref_row.iloc[0]

        for _, row in grp.iterrows():
            if row["Method"] == ref_method:
                continue
            diff_row = row.copy()
            for metric in metrics:
                diff_row[metric] = row[metric] - ref_vals[metric]
            diff_rows.append(diff_row)

    df_diff = pd.DataFrame(diff_rows)
    filtered_methods = [m for m in mapped_methods if m != ref_method]
    df_diff["Method"] = pd.Categorical(
        df_diff["Method"], categories=filtered_methods, ordered=True
    )
    df_diff = df_diff.sort_values(by=["network_id", "Method"]).reset_index(drop=True)

    if output_dir and output_fn:
        summary = (
            df_diff.groupby("Method", observed=False)[metrics]
            .agg(["count", "min", q1, "median", q3, "max", "mean", "std"])
            .stack(future_stack=True)
            .reset_index()
        )
        summary.to_csv(output_dir / f"summary_diff_{output_fn}.csv", index=False)

    fig, axes = plt.subplots(
        nrows=len(metrics),
        ncols=1,
        figsize=(
            len(filtered_methods) * 4 if len(metrics) <= 2 else 12,
            len(metrics) * 5 if len(metrics) <= 2 else len(metrics) * 3,
        ),
        dpi=300,
        tight_layout=True,
        sharex=True,
    )
    if len(metrics) == 1:
        axes = [axes]

    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx]
        ax.grid(True, which="major", axis="y", linestyle="--", color="lightgray")

        sns.boxplot(
            data=df_diff,
            x="Method",
            y=metric,
            ax=ax,
            order=filtered_methods,
            color="white",
            boxprops={"edgecolor": "black", "linewidth": 1.5},
            whiskerprops={"color": "black", "linewidth": 1.5},
            capprops={"color": "black", "linewidth": 1.5},
            medianprops={"color": "red", "linewidth": 1},
            flierprops={
                "marker": "o",
                "markerfacecolor": "black",
                "markeredgecolor": "black",
                "markersize": 3,
            },
        )

        if show_hline:
            kwargs = hline_kwargs or {
                "color": "black",
                "linestyle": "--",
                "linewidth": 1,
                "alpha": 0.5,
            }
            ax.axhline(y=hline_y, **kwargs)

        if ylim:
            ax.set_ylim(*ylim)

        ax.set_ylabel(f"Δ{metric_name}", fontsize=14)
        ax.set_xlabel("Method" if idx == len(metrics) - 1 else "", fontsize=14)

        if "user_time" in metric or "max_mem_usage" in metric:
            ax.set_yscale("log")

        plt.setp(
            ax.get_xticklabels(),
            rotation=90 if len(filtered_methods) > 5 else 0,
            fontsize=14,
        )

    if output_dir and output_fn:
        plt.savefig(output_dir / output_fn, bbox_inches="tight")
        logger.info(f"Saved diff plot to {output_dir / output_fn}")
        plt.close(fig)


# --- Main Execution ---

if __name__ == "__main__":
    args = parse_args()

    log_root = Path(args.log_root)
    acc_root = Path(args.acc_root)
    stats_root = Path(args.stats_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    network_list_path = Path(args.network_list)
    if not network_list_path.exists():
        logger.critical(f"Network list file not found: {network_list_path}")
        sys.exit(1)

    with open(network_list_path, "r") as f:
        network_ids = [line.strip() for line in f.readlines() if line.strip()]

    weights = list(MAPPING.keys())
    csv_path = output_dir / "cd_acc.csv"

    if args.is_load_existing and csv_path.exists():
        logger.info(f"Loading existing data from {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        logger.info("Collecting new data...")
        df = collect_dataframe(
            network_ids,
            weights,
            log_root,
            acc_root,
            stats_root,
            args.generator,
            args.gt_clustering,
            args.gt_id,
            args.n_procs,
        )
        if not df.empty:
            df.to_csv(csv_path, index=False)
            logger.info(f"Data saved to {csv_path}")

    # --- Print Global Completion Summary ---
    if not df.empty:
        print_completion_summary(df)

    # # --- Plotting ---

    # # Density-based methods

    # methods = [
    #     "fista-int",
    #     "fista-int-iter",
    #     "fista-frac-iter",
    #     "flow",
    #     "flow-iter",
    # ]

    # plot_boxplots(
    #     df.copy(),
    #     methods,
    #     metrics=[
    #         "ami",
    #         "ari",
    #         "nmi",
    #     ],
    #     metric_names=[
    #         "AMI",
    #         "ARI",
    #         "NMI",
    #     ],
    #     show_hline=True,
    #     hline_y=0.0,
    #     output_dir=output_dir,
    #     output_fn=f"journal_exp1_clustering.pdf",
    # )

    # plot_boxplots(
    #     df.copy(),
    #     methods,
    #     metrics=[
    #         "f1_score",
    #         "precision",
    #         "recall",
    #     ],
    #     metric_names=[
    #         "F1",
    #         "Precision",
    #         "Recall",
    #     ],
    #     show_hline=True,
    #     hline_y=0.0,
    #     output_dir=output_dir,
    #     output_fn=f"journal_exp1_confusion.pdf",
    # )

    # # No processing

    # methods = [
    #     "flow-iter",
    #     "RTRex",
    #     "leiden-cpm-0.1",
    #     "leiden-cpm-0.01",
    #     "leiden-cpm-0.001",
    #     "leiden-cpm-0.0001",
    #     "leiden-mod",
    #     "infomap+cc",
    #     "ikc-1",
    #     "ikc-2",
    #     "ikc-5",
    #     "ikc-10",
    #     "ikc-20",
    # ]

    # plot_boxplots(
    #     df.copy(),
    #     methods,
    #     metrics=[
    #         "ami",
    #         "ari",
    #         "nmi",
    #     ],
    #     metric_names=[
    #         "AMI",
    #         "ARI",
    #         "NMI",
    #     ],
    #     show_hline=True,
    #     hline_y=0.0,
    #     output_dir=output_dir,
    #     output_fn=f"journal_exp2_clustering.pdf",
    # )

    # plot_boxplots(
    #     df.copy(),
    #     methods,
    #     metrics=[
    #         "f1_score",
    #         "precision",
    #         "recall",
    #     ],
    #     metric_names=[
    #         "F1",
    #         "Precision",
    #         "Recall",
    #     ],
    #     show_hline=True,
    #     hline_y=0.0,
    #     output_dir=output_dir,
    #     output_fn=f"journal_exp2_confusion.pdf",
    # )

    # # Ensembles

    # # Train
    # with open("data/networks_train.txt", "r") as f:
    #     train_network_ids = [line.strip() for line in f.readlines() if line.strip()]

    # # With processing

    # methods = [
    #     # "fista-int",
    #     # "fista-int+wcc",
    #     # "fista-int-iter",
    #     # "fista-int-iter+wcc",
    #     # "fista-frac-iter",
    #     # "fista-frac-iter+wcc",
    #     "flow",
    #     "flow+wcc",
    #     "flow-iter",
    #     "flow-iter+wcc",
    #     "RTRex",
    #     "RTRex+wcc",
    #     "leiden-cpm-0.1",
    #     "leiden-cpm-0.1+wcc",
    #     "leiden-cpm-0.01",
    #     "leiden-cpm-0.01+wcc",
    #     "leiden-cpm-0.001",
    #     "leiden-cpm-0.001+wcc",
    #     "leiden-cpm-0.0001",
    #     "leiden-cpm-0.0001+wcc",
    #     "leiden-mod",
    #     "leiden-mod+wcc",
    #     "infomap+cc",
    #     "infomap+wcc",
    #     "ikc-1",
    #     "ikc-1+wcc",
    #     "ikc-2",
    #     "ikc-2+wcc",
    #     "ikc-5",
    #     "ikc-5+wcc",
    #     "ikc-10",
    #     "ikc-10+wcc",
    #     "ikc-20",
    #     "ikc-20+wcc",
    # ]

    # plot_boxplots(
    #     df.copy(),
    #     methods,
    #     metrics=[
    #         "ami",
    #         "ari",
    #         "nmi",
    #     ],
    #     metric_names=[
    #         "AMI",
    #         "ARI",
    #         "NMI",
    #     ],
    #     show_hline=True,
    #     hline_y=0.0,
    #     output_dir=output_dir,
    #     output_fn=f"journal_exp3_pp_clustering.pdf",
    #     network_list=train_network_ids,
    # )

    # plot_boxplots(
    #     df.copy(),
    #     methods,
    #     metrics=[
    #         "f1_score",
    #         "precision",
    #         "recall",
    #     ],
    #     metric_names=[
    #         "F1",
    #         "Precision",
    #         "Recall",
    #     ],
    #     show_hline=True,
    #     hline_y=0.0,
    #     output_dir=output_dir,
    #     output_fn=f"journal_exp3_pp_confusion.pdf",
    #     network_list=train_network_ids,
    # )

    # # All kinds
    # methods = [
    #     "leiden-cpm-0.01+wcc",
    #     # 2 clusterings
    #     # Fully-weighted
    #     "flow-iter-x-leiden-mod--0-0.0--leiden-cpm-0.01+wcc",
    #     "flow-iter-x-leiden-mod--1-0.0--leiden-cpm-0.01+wcc",
    #     # Majority rule (W)
    #     "flow-iter-x-leiden-mod--0-0.5--leiden-cpm-0.01+wcc",
    #     "flow-iter-x-leiden-mod--1-1--leiden-cpm-0.01+wcc",
    #     # Majority rule (U)
    #     "flow-iter-x-leiden-mod--0-0.5-U--leiden-cpm-0.01+wcc",
    #     "flow-iter-x-leiden-mod--1-1-U--leiden-cpm-0.01+wcc",
    #     # Strict consensus
    #     "flow-iter-x-leiden-mod--0-1.0-U--leiden-cpm-0.01+wcc",
    #     "flow-iter-x-leiden-mod--1-2-U--leiden-cpm-0.01+wcc",
    #     # 3 clusterings
    #     # Fully-weighted
    #     "flow-iter-x-leiden-mod-x-RTRex--0-0.0--leiden-cpm-0.01+wcc",
    #     "flow-iter-x-leiden-mod-x-RTRex--1-0.0--leiden-cpm-0.01+wcc",
    #     # Majority rule (W)
    #     "flow-iter-x-leiden-mod-x-RTRex--0-0.5--leiden-cpm-0.01+wcc",
    #     "flow-iter-x-leiden-mod-x-RTRex--1-2--leiden-cpm-0.01+wcc",
    #     # Majority rule (U)
    #     "flow-iter-x-leiden-mod-x-RTRex--0-0.5-U--leiden-cpm-0.01+wcc",
    #     "flow-iter-x-leiden-mod-x-RTRex--1-2-U--leiden-cpm-0.01+wcc",
    #     # Strict consensus
    #     "flow-iter-x-leiden-mod-x-RTRex--0-1.0-U--leiden-cpm-0.01+wcc",
    #     "flow-iter-x-leiden-mod-x-RTRex--1-3-U--leiden-cpm-0.01+wcc",
    #     # 4 clusterings
    #     # Fully-weighted
    #     "flow-iter-x-leiden-mod-x-RTRex-x-ikc-5--0-0.0--leiden-cpm-0.01+wcc",
    #     "flow-iter-x-leiden-mod-x-RTRex-x-ikc-5--1-0.0--leiden-cpm-0.01+wcc",
    #     # Majority rule (W)
    #     "flow-iter-x-leiden-mod-x-RTRex-x-ikc-5--0-0.5--leiden-cpm-0.01+wcc",
    #     "flow-iter-x-leiden-mod-x-RTRex-x-ikc-5--1-2--leiden-cpm-0.01+wcc",
    #     # Majority rule (U)
    #     "flow-iter-x-leiden-mod-x-RTRex-x-ikc-5--0-0.5-U--leiden-cpm-0.01+wcc",
    #     "flow-iter-x-leiden-mod-x-RTRex-x-ikc-5--1-2-U--leiden-cpm-0.01+wcc",
    #     # Strict majority rule (U)
    #     "flow-iter-x-leiden-mod-x-RTRex-x-ikc-5--0-0.6-U--leiden-cpm-0.01+wcc",
    #     "flow-iter-x-leiden-mod-x-RTRex-x-ikc-5--1-3-U--leiden-cpm-0.01+wcc",
    #     # Strict majority rule (W)
    #     "flow-iter-x-leiden-mod-x-RTRex-x-ikc-5--0-0.6--leiden-cpm-0.01+wcc",
    #     "flow-iter-x-leiden-mod-x-RTRex-x-ikc-5--1-3--leiden-cpm-0.01+wcc",
    #     # Strict consensus
    #     "flow-iter-x-leiden-mod-x-RTRex-x-ikc-5--0-1.0-U--leiden-cpm-0.01+wcc",
    #     "flow-iter-x-leiden-mod-x-RTRex-x-ikc-5--1-4-U--leiden-cpm-0.01+wcc",
    # ]

    # plot_boxplots(
    #     df.copy(),
    #     methods,
    #     metrics=[
    #         "ami",
    #         "ari",
    #         "nmi",
    #     ],
    #     metric_names=[
    #         "AMI",
    #         "ARI",
    #         "NMI",
    #     ],
    #     show_hline=True,
    #     hline_y=0.0,
    #     output_dir=output_dir,
    #     output_fn=f"journal_exp3_train_all_clustering.pdf",
    #     network_list=train_network_ids,
    # )

    # plot_boxplots(
    #     df.copy(),
    #     methods,
    #     metrics=[
    #         "f1_score",
    #         "precision",
    #         "recall",
    #     ],
    #     metric_names=[
    #         "F1",
    #         "Precision",
    #         "Recall",
    #     ],
    #     show_hline=True,
    #     hline_y=0.0,
    #     output_dir=output_dir,
    #     output_fn=f"journal_exp3_train_all_confusion.pdf",
    #     network_list=train_network_ids,
    # )

    logger.info("Processing complete.")

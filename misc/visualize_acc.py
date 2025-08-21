import json
import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

MAPPING = {
    # DSC methods
    "fista-int+cc": "DSC-FISTA(int)",
    "fista-frac+cc": "DSC-FISTA-Iter",
    "flow+cc": "DSC-Flow",
    "flow-iter+cc": "DSC-Flow-Iter",
    "fista-int+wcc": "DSC-FISTA(int)+WCC",
    "fista-frac+wcc": "DSC-FISTA-Iter+WCC",
    "flow+wcc": "DSC-Flow+WCC",
    "flow-iter+wcc": "DSC-Flow-Iter+WCC",
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
    "ikc-1+cc": "IKC(1)",
    "ikc-2+cc": "IKC(2)",
    "ikc-5+cc": "IKC(5)",
    "ikc-10+cc": "IKC(10)",
    "ikc-20+cc": "IKC(20)",
    "ikc-1+wcc": "IKC(1)+WCC",
    "ikc-2+wcc": "IKC(2)+WCC",
    "ikc-5+wcc": "IKC(5)+WCC",
    "ikc-10+wcc": "IKC(10)+WCC",
    "ikc-20+wcc": "IKC(20)+WCC",
    # Fusion methods
    "flow-iter+cc-x-infomap+cc--0.5--leiden-cpm+wcc-0.01": "F+I-M(0.5,W)",
    "flow-iter+cc-x-infomap+cc--0.5(U)--leiden-cpm+wcc-0.01": "F+I-M(0.5,U)",
    "flow-iter+cc-x-infomap+cc--1.0--leiden-cpm+wcc-0.01": "F+I-M(1.0,W)",
    "flow-iter+cc-x-infomap+cc--1.0(U)--leiden-cpm+wcc-0.01": "F+I-M(1.0,U)",
    "flow-iter+cc-x-leiden-mod--0.5--leiden-cpm+wcc-0.01": "F+L-M(0.5,W)",
    "flow-iter+cc-x-leiden-mod--0.5(U)--leiden-cpm+wcc-0.01": "F+L-M(0.5,U)",
    "flow-iter+cc-x-leiden-mod--1.0--leiden-cpm+wcc-0.01": "F+L-M(1.0,W)",
    "flow-iter+cc-x-leiden-mod--1.0(U)--leiden-cpm+wcc-0.01": "F+L-M(1.0,U)",
}

KEEP_FIXED = [
    "fista-int+cc",
    "fista-frac+cc",
    "flow+cc",
    "flow-iter+cc",
    "fista-int+wcc",
    "flow+wcc",
    "fista-frac+wcc",
    "flow-iter+wcc",
    "leiden-mod",
    "leiden-cpm-0.1",
    "leiden-cpm-0.01",
    "leiden-cpm-0.001",
    "leiden-cpm-0.0001",
    "leiden-mod+wcc",
    "leiden-cpm-0.1+wcc",
    "leiden-cpm-0.01+wcc",
    "leiden-cpm-0.001+wcc",
    "leiden-cpm-0.0001+wcc",
    "infomap+cc",
    "infomap+wcc",
    "ikc-10+cc",
    "ikc-10+wcc",
    "ikc-2+cc",
    "ikc-2+wcc",
    "ikc-5+cc",
    "ikc-5+wcc",
]

UPDATE = [
    "flow-iter+cc-x-infomap+cc--0.5--leiden-cpm+wcc-0.01",
    "flow-iter+cc-x-infomap+cc--0.5(U)--leiden-cpm+wcc-0.01",
    "flow-iter+cc-x-infomap+cc--1.0--leiden-cpm+wcc-0.01",
    "flow-iter+cc-x-infomap+cc--1.0(U)--leiden-cpm+wcc-0.01",
    "flow-iter+cc-x-leiden-mod--0.5--leiden-cpm+wcc-0.01",
    "flow-iter+cc-x-leiden-mod--0.5(U)--leiden-cpm+wcc-0.01",
    "flow-iter+cc-x-leiden-mod--1.0--leiden-cpm+wcc-0.01",
    "flow-iter+cc-x-leiden-mod--1.0(U)--leiden-cpm+wcc-0.01",
]


def q1(x):
    return x.quantile(0.25)


def q3(x):
    return x.quantile(0.75)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-root",
        type=str,
        default="data/dsc",
        help="Root directory for the log data",
    )
    parser.add_argument(
        "--acc-root",
        type=str,
        default="data/dsc_acc",
        help="Root directory for the data",
    )
    parser.add_argument(
        "--stats-root",
        type=str,
        default="data/dsc_stats",
        help="Root directory for the stats data",
    )
    parser.add_argument(
        "--network-list",
        type=str,
        default="data/networks_val.txt",
        help="File containing the list of network IDs to visualize",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Directory to save the output plots",
    )
    parser.add_argument(
        "--syn-method",
        type=str,
        default="sbmmcsprev1+o+eL1",
        help="Synthesis method",
    )
    parser.add_argument(
        "--syn-emp-clustering",
        type=str,
        default="sbm_wcc",
        help="Synthesis empirical clustering",
    )
    parser.add_argument(
        "--syn-emp-clustering-res",
        type=str,
        default="sbm",
        help="Synthesis empirical clustering result",
    )
    parser.add_argument(
        "--syn-seed",
        type=str,
        default="0",
        help="Synthesis seed",
    )
    parser.add_argument(
        "--is-load-existing",
        action="store_true",
        default=False,
        help="Load existing data instead of collecting new data",
    )
    parser.add_argument(
        "--n-procs",
        type=int,
        default=8,
        help="Number of processes to use",
    )
    return parser.parse_args()


def read_json_file(path):
    if Path(path).exists():
        with open(path, "r") as f:
            return json.load(f)
    return None


def read_metric_file(
    acc_root,
    syn_method,
    syn_emp_clustering,
    network_id,
    syn_emp_clustering_res,
    syn_seed,
    weight,
    metric_name,
):
    fn = (
        acc_root
        / syn_method
        / syn_emp_clustering
        / network_id
        / syn_emp_clustering_res
        / syn_seed
        / weight
        / f"result.{metric_name}"
    )
    try:
        if Path(fn).exists():
            with open(fn, "r") as f:
                val = f.read().strip()
                return float(val)
    except Exception as e:
        print(f"Error reading {fn}: {e}")
    return None


def process_cluster_connectivity(
    stats_root,
    syn_method,
    syn_emp_clustering,
    network_id,
    syn_emp_clustering_res,
    syn_seed,
    weight,
):
    cluster_connectivity_json = (
        stats_root
        / syn_method
        / syn_emp_clustering
        / network_id
        / syn_emp_clustering_res
        / syn_seed
        / weight
        / "stats.json"
    )
    cluster_connectivity = read_json_file(cluster_connectivity_json)
    result = {}
    if cluster_connectivity:
        result["n_clusters"] = cluster_connectivity["n_clusters"]
        result["n_singleton"] = cluster_connectivity["n_onodes"]

        result["n_disconnected"] = cluster_connectivity["n_disconnects"]
        result["ratio_disconnected"] = (
            result["n_disconnected"] / result["n_clusters"]
            if result["n_clusters"] > 0
            else 0.0
        )

        result["n_wellconnected"] = cluster_connectivity["n_wellconnected_clusters"]
        result["ratio_wellconnected"] = (
            result["n_wellconnected"] / result["n_clusters"]
            if result["n_clusters"] > 0
            else 0.0
        )

        result["n_connected"] = result["n_clusters"] - result["n_disconnected"]
        result["ratio_connected"] = 1.0 - result["ratio_disconnected"]
    else:
        result = {
            "n_clusters": None,
            "n_disconnected": None,
            "ratio_disconnected": None,
            "n_wellconnected": None,
            "ratio_wellconnected": None,
            "n_connected": None,
            "ratio_connected": None,
        }
    return result


def read_log_file(
    log_root,
    syn_method,
    syn_emp_clustering,
    network_id,
    syn_emp_clustering_res,
    syn_seed,
    weight,
):
    def get_log_stats(w):
        log_file = (
            log_root
            / syn_method
            / syn_emp_clustering
            / network_id
            / syn_emp_clustering_res
            / syn_seed
            / w
            / "error.log"
        )
        if not Path(log_file).exists():
            return {"user_time": None, "max_mem_usage": None}
        user_time = None
        mem_usage = None
        with open(log_file, "r") as f:
            for line in f:
                if "User time (seconds):" in line:
                    user_time = float(line.split(":")[-1].strip())
                elif "Maximum resident set size (kbytes):" in line:
                    mem_usage = float(line.split(":")[-1].strip())
        return {"user_time": user_time, "max_mem_usage": mem_usage}

    # Remove +cc/+wcc for base
    if weight.endswith("+cc"):
        base_weight = weight[:-3]
    elif weight.endswith("+wcc"):
        base_weight = weight[:-4]
    else:
        base_weight = weight

    base_stats = get_log_stats(base_weight)
    # Only get processing stats if weight has +cc/+wcc
    if weight.endswith("+cc") or weight.endswith("+wcc"):
        processing_stats = get_log_stats(weight)
    else:
        # No processing necessary, set processing stats to 0
        processing_stats = {"user_time": 0.0, "max_mem_usage": 0.0}

    # Make total time/mem None if any is None
    if base_stats["user_time"] is None or processing_stats["user_time"] is None:
        total_user_time = None
    else:
        total_user_time = base_stats["user_time"] + processing_stats["user_time"]

    if base_stats["max_mem_usage"] is None or processing_stats["max_mem_usage"] is None:
        total_max_mem_usage = None
    else:
        total_max_mem_usage = max(
            base_stats["max_mem_usage"], processing_stats["max_mem_usage"]
        )

    result = {
        "base_user_time": base_stats["user_time"],
        "base_max_mem_usage": base_stats["max_mem_usage"],
        "processing_user_time": processing_stats["user_time"],
        "processing_max_mem_usage": processing_stats["max_mem_usage"],
        "total_user_time": total_user_time,
        "total_max_mem_usage": total_max_mem_usage,
    }
    return result


def _process_task(args):
    """
    Worker function to process a single network_id and weight combination.
    This function handles all file reading and data extraction for one task.
    """
    # Unpack all arguments for clarity
    (
        network_id,
        weight,
        log_root,
        acc_root,
        stats_root,
        syn_method,
        syn_emp_clustering,
        syn_emp_clustering_res,
        syn_seed,
        metrics_list,
    ) = args

    # --- Base Path Definitions ---
    # Define a common base path to avoid rebuilding it repeatedly
    base_path_args = (
        syn_method,
        syn_emp_clustering,
        network_id,
        syn_emp_clustering_res,
        syn_seed,
        weight,
    )
    stats_path = stats_root.joinpath(*base_path_args)
    acc_path = acc_root.joinpath(*base_path_args)

    row_data = {"network_id": network_id, "weight": weight}

    # --- 1. Process Cluster Connectivity ---
    # Use try/except for faster file access (avoids the extra .exists() check)
    try:
        with open(stats_path / "stats.json", "r") as f:
            connectivity = json.load(f)

        n_clusters = connectivity.get("n_clusters", 0)
        n_disconnected = connectivity.get("n_disconnects", 0)

        row_data["n_clusters"] = n_clusters
        row_data["n_singleton"] = connectivity.get("n_onodes")
        row_data["n_disconnected"] = n_disconnected
        row_data["n_wellconnected"] = connectivity.get("n_wellconnected_clusters")

        if n_clusters > 0:
            ratio_disconnected = n_disconnected / n_clusters
            row_data["ratio_disconnected"] = ratio_disconnected
            row_data["ratio_wellconnected"] = (
                row_data.get("n_wellconnected", 0) / n_clusters
            )
            row_data["n_connected"] = n_clusters - n_disconnected
            row_data["ratio_connected"] = 1.0 - ratio_disconnected
        else:
            # Set defaults for division-by-zero cases
            for key in [
                "ratio_disconnected",
                "ratio_wellconnected",
                "n_connected",
                "ratio_connected",
            ]:
                row_data[key] = 0.0

    except (FileNotFoundError, json.JSONDecodeError):
        # If file doesn't exist or is invalid, fill with None
        keys = [
            "n_clusters",
            "n_singleton",
            "n_disconnected",
            "n_wellconnected",
            "ratio_disconnected",
            "ratio_wellconnected",
            "n_connected",
            "ratio_connected",
        ]
        for key in keys:
            row_data[key] = None

    # --- 2. Read Accuracy Metrics ---
    for metric in metrics_list:
        try:
            with open(acc_path / f"result.{metric}", "r") as f:
                row_data[metric] = float(f.read().strip())
                if metric in ["fpr", "fnr"]:
                    row_data[f"comp_{metric}"] = 1.0 - row_data[metric]
        except (FileNotFoundError, ValueError):
            row_data[metric] = None

    # --- 3. Read Log Files for Time and Memory ---
    def get_log_stats(log_path):
        user_time, mem_usage = None, None
        try:
            with open(log_path, "r") as f:
                for line in f:
                    if "User time (seconds):" in line:
                        user_time = float(line.split(":")[-1])
                    elif "Maximum resident set size (kbytes):" in line:
                        mem_usage = float(line.split(":")[-1])
        except FileNotFoundError:
            pass  # Return None, None if file doesn't exist
        return {"user_time": user_time, "max_mem_usage": mem_usage}

    # Determine base and processing log paths
    if weight.endswith(("+cc", "+wcc")):
        suffix_len = 3 if weight.endswith("+cc") else 4
        base_weight = weight[:-suffix_len]
        processing_weight = weight
    else:
        base_weight = weight
        processing_weight = None

    log_base_path_args = (
        syn_method,
        syn_emp_clustering,
        network_id,
        syn_emp_clustering_res,
        syn_seed,
    )
    base_log_path = log_root.joinpath(*log_base_path_args, base_weight, "error.log")
    base_stats = get_log_stats(base_log_path)

    if processing_weight:
        proc_log_path = log_root.joinpath(
            *log_base_path_args, processing_weight, "error.log"
        )
        processing_stats = get_log_stats(proc_log_path)
    else:
        processing_stats = {"user_time": 0.0, "max_mem_usage": 0.0}

    # Combine stats
    row_data["base_user_time"] = base_stats["user_time"]
    row_data["base_max_mem_usage"] = base_stats["max_mem_usage"]
    row_data["processing_user_time"] = processing_stats["user_time"]
    row_data["processing_max_mem_usage"] = processing_stats["max_mem_usage"]

    # Calculate totals, propagating None if any component is missing
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
    network_ids,
    weights,
    log_root,
    acc_root,
    stats_root,
    syn_method,
    syn_emp_clustering,
    syn_emp_clustering_res,
    syn_seed,
    max_workers=8,  # Optional: control number of parallel processes
):
    """
    Collects data by processing files in parallel for significantly faster execution.
    """
    # Convert string paths to Path objects if they aren't already
    log_root = Path(log_root)
    acc_root = Path(acc_root)
    stats_root = Path(stats_root)

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

    # Create a list of all tasks to be executed
    tasks = []
    for network_id in network_ids:
        for weight in weights:
            task_args = (
                network_id,
                weight,
                log_root,
                acc_root,
                stats_root,
                syn_method,
                syn_emp_clustering,
                syn_emp_clustering_res,
                syn_seed,
                metrics_to_read,
            )
            tasks.append(task_args)

    df_data = []
    # Use ProcessPoolExecutor to run tasks in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Use as_completed to process results as they finish, with a tqdm progress bar
        future_to_task = {executor.submit(_process_task, task): task for task in tasks}

        progress_bar = tqdm(
            as_completed(future_to_task), total=len(tasks), desc="Processing Files"
        )
        for future in progress_bar:
            try:
                result = future.result()
                if result:
                    df_data.append(result)
            except Exception as e:
                task_info = future_to_task[future]
                print(
                    f"Error processing task (Net: {task_info[0]}, Weight: {task_info[1]}): {e}"
                )

    if not df_data:
        print("Warning: No data was collected. Check paths and file availability.")
        return pd.DataFrame()

    # Create and finalize the DataFrame
    df = pd.DataFrame(df_data)
    df["weight"] = pd.Categorical(df["weight"], categories=weights, ordered=True)
    df = df.sort_values(by=["network_id", "weight"]).reset_index(drop=True)

    df = df.rename(columns={"weight": "Method"})
    if "MAPPING" in globals() and isinstance(MAPPING, dict):
        df["Method"] = df["Method"].map(MAPPING)

    return df


def plot_boxplots(
    df,
    weights,
    metrics,
    metric_names,
    ylim=None,
    show_hline=False,
    hline_y=0.0,
    hline_kwargs=None,
    output_dir=None,
    output_fn=None,
):
    df = df.dropna(subset=metrics)
    # Only keep rows where Method is in the mapped weights
    mapped_methods = [MAPPING[w] for w in weights if w in MAPPING]
    df = df[df["Method"].isin(mapped_methods)]
    df = df.groupby("network_id").filter(lambda x: len(x) == len(mapped_methods))

    # Generate a summary table for each method
    summary = (
        df.groupby("Method")[metrics]
        .agg(["count", "min", q1, "median", q3, "max", "mean", "std"])
        .stack(future_stack=True)
        .reset_index()
    )
    summary.to_csv(Path(output_dir) / f"summary_{output_fn}.csv", index=False)

    fig, axes = plt.subplots(
        nrows=len(metrics),
        ncols=1,
        figsize=(
            len(mapped_methods) * 2 if len(metrics) <= 2 else 12,
            len(metrics) * 5 if len(metrics) <= 2 else len(metrics) * 3,
        ),
        dpi=300,
        tight_layout=True,
        sharex=True,
    )

    print(
        f"Network IDs ({len(df['network_id'].unique())}): {', '.join(df['network_id'].unique())}"
    )

    # plt.style.use("seaborn-v0_8-whitegrid")

    if len(metrics) == 1:
        axes = [axes]

    # fig.set_facecolor("white")

    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx]  # if len(metrics) > 1 else axes
        ax.grid(True, which="major", axis="y", linestyle="--", color="lightgray")
        sns.boxplot(
            data=df,
            x="Method",
            y=metric,
            ax=ax,
            order=mapped_methods,
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
            ax.axhline(
                y=hline_y,
                **(
                    hline_kwargs
                    if hline_kwargs is not None
                    else {
                        "color": "red",
                        "linestyle": "--",
                        "linewidth": 1,
                        "alpha": 0.7,
                    }
                ),
            )
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_title("")
        ax.set_ylabel(metric_name, fontsize=14)
        if idx == len(metrics) - 1:
            ax.set_xlabel("Method", fontsize=14)
        else:
            ax.set_xlabel("")
        plt.setp(
            ax.get_xticklabels(), rotation=90 if len(weights) > 5 else 0, fontsize=14
        )

        if "user_time" in metric or "max_mem_usage" in metric:
            ax.set_yscale("log")
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("")
    fig.tight_layout()

    if output_dir and output_fn:
        plt.savefig(Path(output_dir) / output_fn, bbox_inches="tight")
        plt.close(fig)


def plot_boxplots_diff(
    df,
    weights,
    ref_weight,
    metrics,
    metric_names,
    ylim=None,
    show_hline=False,
    hline_y=0.0,
    hline_kwargs=None,
    output_dir=None,
    output_fn=None,
):
    df = df.dropna(subset=metrics)
    mapped_methods = [MAPPING[w] for w in weights if w in MAPPING]
    df = df[df["Method"].isin(mapped_methods)]
    df = df.groupby("network_id").filter(lambda x: len(x) == len(mapped_methods))

    print(
        f"Network IDs ({len(df['network_id'].unique())}): {', '.join(df['network_id'].unique())}"
    )

    # Compute difference to reference method for each metric, skip ref_weight in output
    ref_method = MAPPING[ref_weight]
    diff_rows = []
    for network_id in df["network_id"].unique():
        df_net = df[df["network_id"] == network_id]
        ref_row = df_net[df_net["Method"] == ref_method]
        if ref_row.empty:
            continue
        ref_vals = ref_row.iloc[0]
        for _, row in df_net.iterrows():
            if row["Method"] == ref_method:
                continue  # skip ref_weight
            diff_row = row.copy()
            for metric in metrics:
                diff_row[metric] = row[metric] - ref_vals[metric]
            diff_rows.append(diff_row)
    df_diff = pd.DataFrame(diff_rows)
    # Remove ref_method from categories
    filtered_methods = [m for m in mapped_methods if m != ref_method]
    df_diff["Method"] = pd.Categorical(
        df_diff["Method"], categories=filtered_methods, ordered=True
    )
    df_diff = df_diff.sort_values(by=["network_id", "Method"]).reset_index(drop=True)

    # Generate a summary table for each method
    summary = (
        df_diff.groupby("Method")[metrics]
        .agg(["count", "min", q1, "median", q3, "max", "mean", "std"])
        .stack(future_stack=True)
        .reset_index()
    )
    summary.to_csv(Path(output_dir) / f"summary_diff_{output_fn}.csv", index=False)

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
        ax = axes[idx] if len(metrics) > 1 else axes
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
            ax.axhline(
                y=hline_y,
                **(
                    hline_kwargs
                    if hline_kwargs is not None
                    else {
                        "color": "black",
                        "linestyle": "--",
                        "linewidth": 1,
                        "alpha": 0.5,
                    }
                ),
            )
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_title("")
        ax.set_ylabel(f"Δ{metric_name}", fontsize=14)
        if idx == len(metrics) - 1:
            ax.set_xlabel("Method", fontsize=14)
        else:
            ax.set_xlabel("")
        plt.setp(
            ax.get_xticklabels(),
            rotation=90 if len(filtered_methods) > 5 else 0,
            fontsize=14,
        )

        if "user_time" in metric or "max_mem_usage" in metric:
            ax.set_yscale("log")
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("")
    fig.tight_layout()

    if output_dir is not None and output_fn is not None:
        fig.savefig(output_dir / output_fn, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    args = parse_args()

    log_root = Path(args.log_root)
    acc_root = Path(args.acc_root)
    stats_root = Path(args.stats_root)
    network_list = args.network_list
    output_dir = args.output_dir
    n_procs = args.n_procs

    weights = MAPPING.keys()

    with open(network_list, "r") as f:
        network_ids = [line.strip() for line in f.readlines()]

    syn_method = args.syn_method
    syn_emp_clustering = args.syn_emp_clustering
    syn_emp_clustering_res = args.syn_emp_clustering_res
    syn_seed = args.syn_seed
    is_load_existing = args.is_load_existing

    output_dir = Path(f"{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not is_load_existing or not (output_dir / "cd_acc.csv").exists():
        df = collect_dataframe(
            network_ids,
            weights,
            log_root,
            acc_root,
            stats_root,
            syn_method,
            syn_emp_clustering,
            syn_emp_clustering_res,
            syn_seed,
            n_procs,
        )
        df.to_csv(output_dir / "cd_acc.csv", index=False)
    else:
        df = pd.read_csv(output_dir / "cd_acc.csv")

    plot_boxplots(
        df.copy(),
        [
            "fista-int+cc",
            "fista-frac+cc",
            "flow+cc",
            "flow-iter+cc",
        ],
        metrics=[
            "ami",
            "ari",
            "nmi",
        ],
        metric_names=[
            "AMI",
            "ARI",
            "NMI",
        ],
        show_hline=True,
        hline_y=0.0,
        output_dir=output_dir,
        output_fn="exp1.pdf",
    )

    plot_boxplots(
        df.copy(),
        [
            "flow-iter+cc",
            "ikc-1+cc",
            "leiden-cpm-0.01",
            "leiden-mod",
            "infomap+cc",
        ],
        metrics=[
            "ami",
            "ari",
            "nmi",
        ],
        metric_names=[
            "AMI",
            "ARI",
            "NMI",
        ],
        show_hline=True,
        hline_y=0.0,
        output_dir=output_dir,
        output_fn="exp2.pdf",
    )

    plot_boxplots(
        df.copy(),
        [
            "flow-iter+cc",
            "ikc-1+cc",
            "ikc-2+cc",
            "ikc-5+cc",
            "ikc-10+cc",
            "ikc-20+cc",
            "leiden-cpm-0.1",
            "leiden-cpm-0.01",
            "leiden-cpm-0.001",
            "leiden-cpm-0.0001",
            "leiden-mod",
            "infomap+cc",
        ],
        metrics=[
            "ami",
            "ari",
            "nmi",
        ],
        metric_names=[
            "AMI",
            "ARI",
            "NMI",
        ],
        show_hline=True,
        hline_y=0.0,
        output_dir=output_dir,
        output_fn="suppl_exp2.pdf",
    )

    plot_boxplots(
        df.copy(),
        [
            "flow-iter+cc",
            "ikc-1+cc",
            "ikc-2+cc",
            "ikc-5+cc",
            "ikc-10+cc",
            "ikc-20+cc",
            "leiden-cpm-0.1",
            "leiden-cpm-0.01",
            "leiden-cpm-0.001",
            "leiden-cpm-0.0001",
            "leiden-mod",
            "infomap+cc",
        ],
        metrics=[
            "node_coverage",
        ],
        metric_names=[
            "Node Coverage",
        ],
        show_hline=True,
        hline_y=0.0,
        output_dir=output_dir,
        output_fn="suppl_exp2_node_coverage.pdf",
    )

    plot_boxplots(
        df.copy(),
        [
            "leiden-cpm-0.01",
            "flow-iter+wcc",
            "ikc-1+wcc",
            "ikc-2+wcc",
            "ikc-5+wcc",
            "ikc-10+wcc",
            "ikc-20+wcc",
            "leiden-cpm-0.1+wcc",
            "leiden-cpm-0.01+wcc",
            "leiden-cpm-0.001+wcc",
            "leiden-cpm-0.0001+wcc",
            "leiden-mod+wcc",
            "infomap+wcc",
        ],
        metrics=[
            "ami",
            "ari",
            "nmi",
        ],
        metric_names=[
            "AMI",
            "ARI",
            "NMI",
        ],
        show_hline=True,
        hline_y=0.0,
        output_dir=output_dir,
        output_fn="suppl_exp2_wcc.pdf",
    )

    plot_boxplots(
        df.copy(),
        [
            "flow-iter+cc",
            "ikc-1+cc",
            "ikc-2+cc",
            "ikc-5+cc",
            "ikc-10+cc",
            "ikc-20+cc",
            "leiden-cpm-0.1",
            "leiden-cpm-0.01",
            "leiden-cpm-0.001",
            "leiden-cpm-0.0001",
            "leiden-mod",
            "infomap+cc",
        ],
        metrics=[
            "comp_fpr",
            "precision",
            "recall",
        ],
        metric_names=[
            "1 - FPR",
            "Precision",
            "Recall",
        ],
        show_hline=True,
        hline_y=0.0,
        output_dir=output_dir,
        output_fn="suppl_exp3_confusion.pdf",
    )

    plot_boxplots(
        df.copy(),
        [
            "flow-iter+cc",
            "ikc-1+cc",
            "leiden-cpm-0.01",
            "leiden-mod",
            "infomap+cc",
        ],
        metrics=[
            "comp_fpr",
            "precision",
            "recall",
        ],
        metric_names=[
            "1 - FPR",
            "Precision",
            "Recall",
        ],
        show_hline=True,
        hline_y=0.0,
        output_dir=output_dir,
        output_fn="exp3_confusion.pdf",
    )

    plot_boxplots_diff(
        df.copy(),
        [
            "leiden-cpm-0.01+wcc",
            "flow-iter+cc-x-leiden-mod--0.5--leiden-cpm+wcc-0.01",
            "flow-iter+cc-x-leiden-mod--0.5(U)--leiden-cpm+wcc-0.01",
            "flow-iter+cc-x-leiden-mod--1.0--leiden-cpm+wcc-0.01",
            "flow-iter+cc-x-leiden-mod--1.0(U)--leiden-cpm+wcc-0.01",
            "flow-iter+cc-x-infomap+cc--0.5--leiden-cpm+wcc-0.01",
            "flow-iter+cc-x-infomap+cc--0.5(U)--leiden-cpm+wcc-0.01",
            "flow-iter+cc-x-infomap+cc--1.0--leiden-cpm+wcc-0.01",
            "flow-iter+cc-x-infomap+cc--1.0(U)--leiden-cpm+wcc-0.01",
        ],
        "leiden-cpm-0.01+wcc",
        metrics=[
            "ami",
            "ari",
            "nmi",
        ],
        metric_names=[
            "AMI",
            "ARI",
            "NMI",
        ],
        show_hline=True,
        hline_y=0.0,
        output_dir=output_dir,
        output_fn="suppl_exp3_leiden_fil_u_diff.pdf",
    )

    plot_boxplots_diff(
        df.copy(),
        [
            "leiden-cpm-0.01+wcc",
            "flow-iter+cc-x-leiden-mod--0.5--leiden-cpm+wcc-0.01",
            "flow-iter+cc-x-leiden-mod--0.5(U)--leiden-cpm+wcc-0.01",
            "flow-iter+cc-x-leiden-mod--1.0--leiden-cpm+wcc-0.01",
            "flow-iter+cc-x-leiden-mod--1.0(U)--leiden-cpm+wcc-0.01",
        ],
        "leiden-cpm-0.01+wcc",
        metrics=[
            "ami",
            "ari",
            "nmi",
        ],
        metric_names=[
            "AMI",
            "ARI",
            "NMI",
        ],
        show_hline=True,
        hline_y=0.0,
        output_dir=output_dir,
        output_fn="suppl_exp3_leiden_fil_u_diff.pdf",
    )

    plot_boxplots(
        df.copy(),
        [
            "flow-iter+cc",
            "leiden-mod",
            "leiden-cpm-0.01+wcc",
            "flow-iter+cc-x-leiden-mod--0.5(U)--leiden-cpm+wcc-0.01",
        ],
        metrics=[
            "ami",
            "ari",
            "nmi",
        ],
        metric_names=[
            "AMI",
            "ARI",
            "NMI",
        ],
        show_hline=True,
        hline_y=0.0,
        output_dir=output_dir,
        output_fn="exp3_fl_leiden.pdf",
    )

# for each nw_id from data/networks_val.txt
# find all stage_id that data/dsc_real/<nw_id>/<stage_id>/error.log exists
# check if "Command terminated by signal <signal>" has signal 0
# if not, then return None
# else, parse the line "Elapsed (wall clock) time (h:mm:ss or m:ss): <time>" to get time
# if found, then return time
# else, return None
# store in a pandas DataFrame
import os
from pathlib import Path
import re

import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt


def parse_time(s):
    # if it is in the format h:mm:ss
    match = re.match(r"(\d+):(\d+):(\d+)", s)
    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        seconds = int(match.group(3))
        return hours * 3600 + minutes * 60 + seconds
    # if it is in the format m:ss.ff
    match = re.match(r"(\d+):(\d+)(?:\.(\d+))?", s)
    if match:
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        frac_seconds = float("0." + match.group(3)) if match.group(3) else 0.0
        return minutes * 60 + seconds + frac_seconds
    return None


def parse_time_from_log(log_file):
    with open(log_file, "r") as f:
        for line in f:
            if "Command terminated by signal" in line:
                match = re.search(r"Command terminated by signal (\d+)", line)
                if match:
                    signal = match.group(1)
                    if not signal == "0":
                        return None

            if "Elapsed (wall clock) time" in line:
                match = re.search(
                    r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", line
                )
                if match:
                    return parse_time(match.group(1))
    return None


nw_ids = []
with open("data/networks_val.txt") as f:
    for line in f:
        nw_ids.append(line.strip())

df_data = []
all_stages = set()

for nw_id in nw_ids:
    if os.path.exists(f"data/dsc_real/{nw_id}"):
        for stage_id in os.listdir(f"data/dsc_real/{nw_id}"):
            all_stages.add(stage_id)

for nw_id in nw_ids:
    stages_data = {stage_id: None for stage_id in all_stages}
    stages_data["network_id"] = nw_id

    if os.path.exists(f"data/dsc_real/{nw_id}"):
        for stage_id in os.listdir(f"data/dsc_real/{nw_id}"):
            log_file = f"data/dsc_real/{nw_id}/{stage_id}/error.log"
            if os.path.exists(log_file):
                time = parse_time_from_log(log_file)
                stages_data[stage_id] = time

    df_data.append(stages_data)

df = pd.DataFrame(df_data)
df["stage_1"] = df["dsc-flow-iter"]
df["stage_2"] = df["leiden-mod"]
df["stage_3"] = df["merged"] + df["unweighted"]
df["stage_4"] = df["final"] + df["wcc"]

df["total_time"] = df[["stage_1", "stage_2", "stage_3", "stage_4"]].sum(axis=1)
df["stage_1_perc"] = df["stage_1"] / df["total_time"] * 100
df["stage_2_perc"] = df["stage_2"] / df["total_time"] * 100
df["stage_3_perc"] = df["stage_3"] / df["total_time"] * 100
df["stage_4_perc"] = df["stage_4"] / df["total_time"] * 100

df = df.round(3)

network_stats = pd.read_csv("data/network_stats.csv")
df = df.merge(network_stats, on="network_id", how="left")

df = df[
    [
        "network_id",
        "n_nodes",
        "n_edges",
        "stage_1",
        "stage_2",
        "stage_3",
        "stage_4",
        "total_time",
        "stage_1_perc",
        "stage_2_perc",
        "stage_3_perc",
        "stage_4_perc",
    ]
]
df = df.sort_values(by="total_time")

out_dir = Path("plots/sbm_wcc/sbm/time")
out_dir.mkdir(parents=True, exist_ok=True)

df.to_csv(out_dir / "results.csv", index=False)


def get_representative_networks(df, user_specified_networks=None):
    if user_specified_networks:
        representative_networks = df[
            df["network_id"].isin(user_specified_networks)
        ].to_dict(orient="records")
    else:
        gmm = GaussianMixture(n_components=5, random_state=42)
        df = df.dropna(
            subset=["stage_1_perc", "stage_2_perc", "stage_3_perc", "stage_4_perc"]
        )
        X = df[["stage_1_perc", "stage_2_perc", "stage_3_perc", "stage_4_perc"]]
        gmm.fit(X)

        labels = gmm.predict(X)
        unique_labels = set(labels)

        representative_networks = []
        if unique_labels:
            for label in unique_labels:
                cluster_data = df.loc[labels == label]
                representative_network = cluster_data.iloc[
                    -1
                ]  # Pick the last network in the cluster
                representative_networks.append(representative_network)

    # Sort representative networks by n_nodes
    return sorted(representative_networks, key=lambda x: x["n_nodes"])


def plot_runtime_breakdown(representative_networks, output_dir, output_fn):
    # Create a stacked bar plot
    fig, ax = plt.subplots(figsize=(17, 7))
    for i, network in enumerate(representative_networks):
        ax.bar(
            [i],
            network["stage_1"] / 60,  # Convert to minutes
            label="Stage 1" if i == 0 else "_nolegend_",
            color="blue",
        )
        ax.bar(
            [i],
            network["stage_2"] / 60,  # Convert to minutes
            bottom=network["stage_1"] / 60,
            label="Stage 2" if i == 0 else "_nolegend_",
            color="orange",
        )
        ax.bar(
            [i],
            network["stage_3"] / 60,  # Convert to minutes
            bottom=(network["stage_1"] + network["stage_2"]) / 60,
            label="Stage 3" if i == 0 else "_nolegend_",
            color="green",
        )
        ax.bar(
            [i],
            network["stage_4"] / 60,  # Convert to minutes
            bottom=(network["stage_1"] + network["stage_2"] + network["stage_3"]) / 60,
            label="Stage 4" if i == 0 else "_nolegend_",
            color="red",
        )
        # Add total running time to the top of the bar
        total_time_minutes = int(network["total_time"] // 60)  # Minutes part
        total_time_seconds = int(network["total_time"] % 60)  # Seconds part
        ax.text(
            i,
            (network["total_time"] / 60),  # Position above the bar
            f"{total_time_minutes}m{total_time_seconds:02d}s",
            ha="center",
            va="bottom",
            fontsize=14,
            color="black",
        )

    ax.set_xticks(range(len(representative_networks)))
    ax.set_xticklabels(
        [
            f"{network['network_id']}\n|V| = {network['n_nodes']:,}\n|E| = {network['n_edges']:,}"
            for network in representative_networks
        ],
        fontsize=16,  # Further increase font size for x-axis labels
    )
    ax.set_ylabel(
        "Total Time (minutes)", fontsize=18
    )  # Update label to reflect minutes
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4, fontsize=16
    )  # Further increase font size for legend
    plt.tight_layout()
    plt.savefig(output_dir / output_fn, dpi=300)  # Set DPI to 300


plot_runtime_breakdown(
    get_representative_networks(
        df,
        [
            "citeseer",
            "petster",
            "yahoo_ads",
            "berkstan_web",
            "myspace_aminer",
            "hyves",
        ],
    ),
    out_dir,
    "runtime_breakdown_large.pdf",
)

plot_runtime_breakdown(
    get_representative_networks(
        df,
        [
            "uni_email",
            "fediverse",
            "sp_infectious",
            "twitter",
        ],
    ),
    out_dir,
    "runtime_breakdown_small.pdf",
)

# fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=False, sharey=True)

# Define stages and their corresponding data
stages = [
    ("Total", "total_time"),
    ("Stage 1", "stage_1"),
    ("Stage 2+3+4", ["stage_2", "stage_3", "stage_4"]),
]

# Define groups based on total_time
groups = [
    ("Total $<$ 10s", df["total_time"] < 10),
    ("Total $\\in$ [10s, 100s)", (df["total_time"] >= 10) & (df["total_time"] < 100)),
    (
        "Total $\\in$ [100s, 1000s)",
        (df["total_time"] >= 100) & (df["total_time"] < 1000),
    ),
    ("Total $\\geq$ 1000s", df["total_time"] >= 1000),
]

use_log_scale = True  # Set this to False if log-scale is not desired
include_second_row = False  # Set this to True to include the second row
use_color_grouping = False  # Set this to False to disable color grouping

# Adjust figure size based on whether the second row is included
if include_second_row:
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=False, sharey=True)
else:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=False, sharey=True)

# Top row: Node-based plots
for i, (title, column) in enumerate(stages):
    ax = axes[0, i] if include_second_row else axes[i]
    for j, (group_label, group_filter) in enumerate(groups):
        if isinstance(
            column, list
        ):  # If it's the rest time (sum of stages 2, 3, and 4)
            data = df.loc[group_filter, column].sum(axis=1)
        else:
            data = df.loc[group_filter, column]
        color = f"C{j}" if use_color_grouping else "blue"
        ax.scatter(
            df.loc[group_filter, "n_nodes"] / (1e6 if not use_log_scale else 1),
            data / (60 if not use_log_scale else 1),
            c=color,
            label=group_label if use_color_grouping else None,
            s=8,
        )
    # Add horizontal, thin, dotted, red lines at runtime = 10, 100, 1000 seconds
    for runtime in [10, 100, 1000]:
        ax.axhline(
            y=runtime / (60 if not use_log_scale else 1),
            color="red",
            linestyle="dashed",
            linewidth=0.3,
            alpha=0.5,
        )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(
        "Number of Nodes (millions)" if not use_log_scale else "Number of Nodes",
        fontsize=14,
    )
    if use_log_scale:
        ax.set_xscale("log")  # Set x-axis to log scale
        ax.set_yscale("log")  # Set y-axis to log scale
    if i == 0:
        ax.set_ylabel(
            "Runtime (minutes)" if not use_log_scale else "Runtime (seconds)",
            fontsize=14,
        )
    # ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)  # Add grid

if include_second_row:
    # Bottom row: Edge-based plots
    for i, (title, column) in enumerate(stages):
        ax = axes[1, i]
        for j, (group_label, group_filter) in enumerate(groups):
            if isinstance(
                column, list
            ):  # If it's the rest time (sum of stages 2, 3, and 4)
                data = df.loc[group_filter, column].sum(axis=1)
            else:
                data = df.loc[group_filter, column]
            color = f"C{j}" if use_color_grouping else "blue"
            ax.scatter(
                df.loc[group_filter, "n_edges"] / (1e6 if not use_log_scale else 1),
                data / (60 if not use_log_scale else 1),
                c=color,
                label=group_label if use_color_grouping and i == 0 else None,
                s=8,
            )
        ax.set_xlabel(
            "Number of Edges (millions)" if not use_log_scale else "Number of Edges",
            fontsize=14,
        )
        if use_log_scale:
            ax.set_xscale("log")  # Set x-axis to log scale
            ax.set_yscale("log")  # Set y-axis to log scale
        if i == 0:
            ax.set_ylabel(
                "Runtime (minutes)" if not use_log_scale else "Runtime (seconds)",
                fontsize=14,
            )

if use_color_grouping:
    plt.legend(loc="upper right", fontsize=10)
plt.tight_layout()
if use_log_scale:
    if include_second_row and use_color_grouping:
        plt.savefig(
            out_dir / "log_scale_scatter_plots_with_second_row_and_color.pdf",
            dpi=300,
            bbox_inches="tight",
        )
    elif include_second_row:
        plt.savefig(
            out_dir / "log_scale_scatter_plots_with_second_row.pdf",
            dpi=300,
            bbox_inches="tight",
        )
    elif use_color_grouping:
        plt.savefig(
            out_dir / "log_scale_scatter_plots_with_color.pdf",
            dpi=300,
            bbox_inches="tight",
        )
    else:
        plt.savefig(
            out_dir / "log_scale_scatter_plots.pdf", dpi=300, bbox_inches="tight"
        )
else:
    if include_second_row and use_color_grouping:
        plt.savefig(
            out_dir / "scatter_plots_with_second_row_and_color.pdf",
            dpi=300,
            bbox_inches="tight",
        )
    elif include_second_row:
        plt.savefig(
            out_dir / "scatter_plots_with_second_row.pdf", dpi=300, bbox_inches="tight"
        )
    elif use_color_grouping:
        plt.savefig(
            out_dir / "scatter_plots_with_color.pdf", dpi=300, bbox_inches="tight"
        )
    else:
        plt.savefig(out_dir / "scatter_plots.pdf", dpi=300, bbox_inches="tight")

import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
import argparse


def detect_delimiter(file_path):
    with open(file_path, "r") as f:
        first_line = f.readline().strip()
        if "\t" in first_line:
            return "\t"
        elif "," in first_line:
            return ","
        elif " " in first_line:
            return " "
        else:
            raise ValueError("Unsupported delimiter in input file.")
    return ","


parser = argparse.ArgumentParser(description="Plot density scatter plot from TSV file.")
parser.add_argument(
    "--input-1", type=Path, required=True, help="Input TSV file with density data"
)
parser.add_argument(
    "--aux-input-1",
    type=Path,
    default=None,
    help="Optional TSV file for the first input",
)
parser.add_argument(
    "--input-2", type=Path, default=None, help="Input TSV file with density data"
)
parser.add_argument(
    "--prefix-output", type=Path, required=True, help="Output prefix for plot files"
)
args = parser.parse_args()

if args.input_2 is None:
    inp1 = args.input_1
    aux_inp1 = args.aux_input_1
    prefix_out = args.prefix_output

    prefix_out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(12, 4), sharey=True)

    delimiter = detect_delimiter(inp1)
    df1 = pd.read_csv(inp1, sep=delimiter, names=["node_id", "density"])
    df1_sorted = df1.sort_values(by="density")

    ax.scatter(
        range(len(df1_sorted)), df1_sorted["density"], s=1, alpha=0.5, label="FISTA"
    )
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.set_ylim(0)
    ax.grid(True)

    if aux_inp1:
        delimiter = detect_delimiter(aux_inp1)
        df1_aux = pd.read_csv(aux_inp1, sep=delimiter, names=["node_id", "density"])
        df1_aux_sorted = df1_aux.sort_values(by="density")
        ax.scatter(
            range(len(df1_aux_sorted)),
            df1_aux_sorted["density"],
            s=1,
            alpha=0.5,
            color="orange",
            label="FISTA(int)",
        )

    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(str(prefix_out) + ".png", dpi=300)
else:
    inp1 = args.input_1
    aux_inp1 = args.aux_input_1
    inp2 = args.input_2
    prefix_out = args.prefix_output

    prefix_out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Plot for input_1
    delimiter = detect_delimiter(inp1)
    df1 = pd.read_csv(inp1, sep=delimiter, names=["node_id", "density"])
    df1_sorted = df1.sort_values(by="density")

    # Use the order of node_id from df1_sorted as the reference
    reference_order = df1_sorted["node_id"].tolist()

    ax.scatter(
        range(len(df1_sorted)),
        df1_sorted["density"],
        s=1,
        alpha=1.0,
        label="FISTA",
        rasterized=False,
    )

    if aux_inp1:
        delimiter = detect_delimiter(aux_inp1)
        df1_aux = pd.read_csv(aux_inp1, sep=delimiter, names=["node_id", "density"])
        df1_aux = df1_aux.set_index("node_id").reindex(reference_order).reset_index()
        ax.scatter(
            range(len(df1_aux)),
            df1_aux["density"],
            s=1,
            alpha=0.5,
            color="orange",
            label="FISTA(int)",
            rasterized=False,
        )

    # Plot for input_2
    delimiter = detect_delimiter(inp2)
    df2 = pd.read_csv(inp2, sep=delimiter, names=["node_id", "density"])
    df2 = df2.set_index("node_id").reindex(reference_order).reset_index()
    ax.scatter(
        range(len(df2)),
        df2["density"],
        s=1,
        alpha=1.0,
        color="red",
        label="Flow",
        rasterized=False,
    )

    ax.set_xlabel("Index", fontsize=18)
    ax.set_ylabel("Value", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylim(0)
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=16)

    fig.tight_layout()
    fig.savefig(str(prefix_out) + ".pdf", dpi=300)

# # Cluster the density values
# from sklearn.cluster import DBSCAN
# import numpy as np

# # Reshape density values for clustering
# X = df_sorted[['density']].values

# # Use DBSCAN for clustering without specifying number of clusters
# dbscan = DBSCAN(eps=0.05, min_samples=5)
# df_sorted['cluster'] = dbscan.fit_predict(X)
# # Plot the clusters
# plt.figure(figsize=(10, 6))
# plt.scatter(range(len(df_sorted)), df_sorted['density'], c=df_sorted['cluster'], cmap='viridis', s=1, alpha=0.5)
# plt.title('Clusters of Density Values')
# plt.xlabel('Density')
# plt.ylabel('Index')
# plt.colorbar(label='Cluster')
# plt.grid(True)
# plt.savefig('clustering_plot.png', dpi=300)

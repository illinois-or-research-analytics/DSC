import sys
import time
import logging
import argparse
from pathlib import Path

import igraph as ig
import pandas as pd
import leidenalg as la


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--edgelist",
        type=str,
        required=True,
        help="Path to the edgelist file",
    )
    parser.add_argument(
        "--output-directory",
        type=str,
        required=True,
        help="Directory to save the output files",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["cpm", "mod"],
        help="Model to use for clustering",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        help="Resolution parameter for the CPM model (default: None)",
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility (default: 123456)",
        default=123456,
    )
    parser.add_argument(
        "--n-iterations",
        type=int,
        help="Number of iterations for the Leiden algorithm (default: 2)",
        default=2,
    )
    return parser.parse_args()


args = parse_args()
edgelist_fn = args.edgelist
output_dir = Path(args.output_directory)
model = args.model
resolution = args.resolution
seed = args.seed
n_iterations = args.n_iterations

# ===========

output_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=output_dir / "run.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

# ===========

start = time.perf_counter()

df = pd.read_csv(edgelist_fn, dtype=str, sep="\t")
g = ig.Graph.TupleList(
    df.itertuples(index=False), directed=False, vertex_name_attr="name"
)

elapsed = time.perf_counter() - start
logging.info(f"[TIME] Loading network: {elapsed}")

# ===========

start = time.perf_counter()

if model == "cpm":
    partition = la.find_partition(
        g,
        la.CPMVertexPartition,
        resolution_parameter=resolution,
        seed=seed,
        n_iterations=n_iterations,
    )
elif model == "mod":
    partition = la.find_partition(
        g, la.ModularityVertexPartition, seed=seed, n_iterations=n_iterations
    )
else:
    raise ValueError(f"Unknown model: {model}")

elapsed = time.perf_counter() - start
logging.info(f"[TIME] Running Leiden algorithm: {elapsed}")

# ===========

start = time.perf_counter()

df2 = pd.DataFrame(
    {
        "node_id": [g.vs[i]["name"] for i in g.vs.indices],
        "cluster_id": partition.membership,
    }
)
df2.to_csv(output_dir / "com.tsv", index=False, sep="\t", header=False)

elapsed = time.perf_counter() - start
logging.info(f"[TIME] Saving results: {elapsed}")

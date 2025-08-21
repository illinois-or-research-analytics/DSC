import sys
import time
import logging
import argparse
from pathlib import Path

from infomap import Infomap


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
    args = parser.parse_args()
    return args


args = parse_args()

args = parse_args()
edgelist_fn = args.edgelist
output_dir = Path(args.output_directory)

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

im = Infomap()
with open(edgelist_fn) as f:
    for line in f:
        u, v = line.split()
        im.add_link(int(u), int(v))

elapsed = time.perf_counter() - start
logging.info(f"[TIME] Loading network: {elapsed}")

# ===========

start = time.perf_counter()

im.run()

elapsed = time.perf_counter() - start
logging.info(f"[TIME] Running Infomap algorithm: {elapsed}")

# ===========

start = time.perf_counter()

cluster_dict = {}
for node in im.tree:
    if node.is_leaf:
        if node.module_id not in cluster_dict:
            cluster_dict[node.module_id] = []
        cluster_dict[node.module_id].append(node.node_id)

with open(output_dir / "com.tsv", "w") as f:
    for key in cluster_dict:
        for val in cluster_dict[key]:
            f.write(f"{val}\t{key}\n")

elapsed = time.perf_counter() - start
logging.info(f"[TIME] Saving results: {elapsed}")

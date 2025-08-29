# Dense Subgraph Clustering

## Overview

The repository contains the implementation of density-based community detection methods, including the recommended DSC-Flow-Iter and other methods such as DSC-FISTA(int), DSC-Flow, and DSC-FISTA-Iter. 

The repository also contains the script to run a recommended pipeline, which consists of four stages:
1. Running DSC-Flow-Iter on the input network
2. Running Leiden-Mod on the input network
3. Combining the clusterings from Stage 1 and 2 to obtain a network using the cluster ensemble technique
4. Running Leiden-CPM(0.01) on the obtained network from Stage 3 and post-processing the result with WCC.

## Usage

### Running a density-based technique

**Command** We can run DSC-Flow-Iter using the following command:
```bash
./bin/flow-iter <edgelist> <com> <density>
```
where
- `<edgelist>` is the path to the input edgelist file (TSV format)
- `<com>` is the path to the output community file (TSV format)
- `<density>` is the path to the output density file (TSV format)

For DSC-Flow, replace `./bin/flow-iter` with `./bin/flow`.

We can run DSC-FISTA(int) using the following command:

```bash
./bin/fista-int <niters> <edgelist> <com> <density>
```
where
- `<niters>` is the number of iterations to run (recommended: 200)
- `<edgelist>` is the path to the input edgelist file (TSV format)
- `<com>` is the path to the output community file (TSV format)
- `<density>` is the path to the output density file (TSV format)

For DSC-FISTA-Iter, replace `./bin/fista-int` with `./bin/fista-frac`.

Please make sure the parent directory of `<com>` and `<density>` exists before running the command. Otherwise, it will still run without producing the output files.

**Example**

```bash
./bin/flow-iter test/input/bitcoin_alpha.tsv test/output/dsc-flow-iter/bitcoin_alpha/com.tsv test/output/dsc-flow-iter/bitcoin_alpha/density.tsv
```

```bash
./bin/fista-int 200 test/input/bitcoin_alpha.tsv test/output/dsc-fista-int/bitcoin_alpha/com.tsv test/output/dsc-fista-int/bitcoin_alpha/density.tsv
```

### Running the recommended pipeline

**Command** We can run the recommended pipeline using the following command:
```bash
bash pipeline.sh <edgelist> <output_directory>
```
where
- `<edgelist>` is the path to the input edgelist file (TSV format)
- `<output_directory>` is the path to the output directory where the results will be saved

**Output**

The output will be saved in the specified `<output_directory>`. The main results are:
- Stage 1:
    - `dsc-flow-iter/com.tsv`: The community detection result of DSC-Flow-Iter
    - `dsc-flow-iter+cc/com.tsv`: The community detection result after post-processing with CC
- Stage 2:
    - `leiden-mod/com.tsv`: The community detection result of Leiden-Mod
- Stage 3:
    - `merged/edge.tsv`: The edgelist of the network obtained from combining the results of Stage 1 and 2. This will be a weighted network.
    - `unweighted/com.tsv`: The edgelist of the network without weights obtained by removing the weights from the weighted network.
- Stage 4:
    - `final/com.tsv`: The community detection result obtained by running Leiden-CPM(0.01) on the merged network.
    - `wcc/com.tsv`: The community detection result after post-processing the final result with WCC.

Hence, the output community detection results will be available in `<output_directory>/wcc/com.tsv`.

## Installation

## Setup external dependencies

Setup `ClusterMerger` for combining clusterings (require cmake, bison, flex)
```bash
cd ClusterMerger
./setup.sh
./easy_build_and_compile.sh
```

Setup `constrained-clustering` for post-processing with CC and WCC (require cmake, bison, flex)
```bash
cd constrained-clustering
./setup.sh
./easy_build_and_compile.sh
```

### Running a density based technique

To build and compile density-based methods, run the following command:
```
bash build.sh
```

Note that it is recommended to post-process the result with CC.

### Running the recommended pipeline

To install additional Python dependencies, run:
```bash
pip install leidenalg pandas
```
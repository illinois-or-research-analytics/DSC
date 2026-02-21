# Dense Subgraph Clustering

## Overview

The repository contains the implementation of density-based community detection methods, including the recommended DSC-Flow-Iter and other methods such as DSC-FISTA(int), DSC-Flow, and DSC-FISTA-Iter. 

The repository also contains the script to run a recommended pipeline, which consists of four stages:
1. Running DSC-Flow-Iter on the input network
2. Running Leiden-Mod, RTRex, IKC(5) on the input network
3. Constructing an unweighted consensus network using the constrained voting strategy at the majority rule consensus level
4. Running Leiden-CPM(0.01) on the obtained network from Stage 3 and post-processing the result with WCC.

The preprint of the conference version of the work (which described a different pipeline) and supplementary materials are on the arxiv ([link](https://arxiv.org/abs/2508.17013v2)). If you use our work, you can use the following BibTeX entry to cite.
```
@misc{vule2025dsc-conf-arxiv,
      title={Dense Subgraph Clustering and a New Cluster Ensemble Method}, 
      author={The-Anh Vu-Le and João Alfredo Cardoso Lamy and Tomás Alessi and Ian Chen and Minhyuk Park and Elfarouk Harb and George Chacko and Tandy Warnow},
      year={2025},
      eprint={2508.17013},
      archivePrefix={arXiv},
      primaryClass={cs.SI},
      url={https://arxiv.org/abs/2508.17013v2}, 
}
```
The extended version of the work with the new recommended pipeline (implemented here) is being prepared for journal submission. We will update the README with the link to the preprint of the journal version once it is available.

## Usage

### Running a DSC technique

**Command** We can run *DSC-Flow-Iter* using the following command:
```bash
./bin/flow-iter <edgelist> <com> <density>
```
where
- `<edgelist>` is the path to the input edgelist file (CSV format with header `source,target`)
- `<com>` is the path to the output community file (CSV format with header `node_id,cluster_id`)
- `<density>` is the path to the output density file (CSV format with header `node_id,value`)

For *DSC-Flow*, replace `./bin/flow-iter` with `./bin/flow`.

We can run *DSC-FISTA(int)-Iter* using the following command:

```bash
./bin/fista-int-iter <niters> <edgelist> <com> <density>
```
where
- `<niters>` is the number of iterations to run (recommended: 200)
- `<edgelist>` is the path to the input edgelist file (CSV format with header `source,target`)
- `<com>` is the path to the output community file (CSV format with header `node_id,cluster_id`)
- `<density>` is the path to the output density file (CSV format with header `node_id,value`)

For *DSC-FISTA-Iter*, replace `./bin/fista-int-iter` with `./bin/fista-frac-iter`. For *DSC-FISTA(int)*, replace `./bin/fista-int-iter` with `./bin/fista-int`.

**Note** Please make sure the parent directory of `<com>` and `<density>` exists before running the command. Otherwise, it will still run without producing the output files.

**Example**

```bash
./bin/flow-iter test/input/bitcoin_alpha.csv test/output/dsc-flow-iter/bitcoin_alpha/com.csv test/output/dsc-flow-iter/bitcoin_alpha/density.csv
```

```bash
./bin/fista-int-iter 200 test/input/bitcoin_alpha.csv test/output/dsc-fista-int/bitcoin_alpha/com.csv test/output/dsc-fista-int/bitcoin_alpha/density.csv
```

### Running the recommended pipeline

**Command** We can run the recommended pipeline using the following command:
```bash
bash pipeline.sh <edgelist> <output_directory>
```
where
- `<edgelist>` is the path to the input edgelist file (CSV format with header `source,target`)
- `<output_directory>` is the path to the output directory where the results will be saved

**Output**

The output will be saved in the specified `<output_directory>`. The main results are:
- Stage 1:
    - `dsc-flow-iter/com.csv`: The community detection result of DSC-Flow-Iter
- Stage 2:
    - `leiden-mod/com.csv`: The community detection result of Leiden-Mod
    - `RTRex/com.csv`: The community detection result of RTRex
    - `ikc-5/com.csv`: The community detection result of IKC(5)
- Stage 3:
    - `merged/edge.csv`: The edgelist of the network obtained from combining the results of Stage 1 and 2. This will be a weighted network.
    - `unweighted/edge.csv`: The edgelist of the network without weights obtained by removing the weights from the weighted network.
- Stage 4:
    - `final/com.csv`: The community detection result obtained by running Leiden-CPM(0.01) on the merged network.
    - `final+wcc/com.csv`: The community detection result after post-processing the final result with WCC.

Hence, the output community detection results will be available in `<output_directory>/final+wcc/com.csv`.

## Installation

## Setup external dependencies

### DSC methods

To build and compile DSC methods, run the following command:
```
bash build.sh
```

### Python dependencies

To install additional Python dependencies, run:
```bash
pip install leidenalg networkit pandas
```

`leidenalg` and `networkit` are required to run Leiden algorithms and IKC. `pandas` is the common dependency for all methods to process CSV files.

### RTRex

Make edits to `amazon-RTRExtractor/RTRex/Escape/Nucleus.h` to fix a warning: replace `free(stack)` with `delete[] stack` in line 241 and line 301. This is because stack is allocated using `new[]`, so it should be deallocated using `delete[]` instead of `free()`. This is usually a warning, but RTRex is compiled with `-Werror`, which treats warnings as errors, so it will fail to compile without this fix.

Then, setup RTRex using the following commands from the root directory of the repository:
```bash
cd amazon-RTRExtractor/RTRex/clustering
make clean
cd ..
make clean
make
```

The binary file will be available at `amazon-RTRExtractor/RTRex/clustering/RTRex`. It is recommended to move the binary file to `bin/RTRex` to run the wrapper script (i.e., run `mv amazon-RTRExtractor/RTRex/clustering/RTRex bin/RTRex`). Also make sure to give execute permission to the binary file (i.e., run `chmod +x bin/RTRex`).

### Cluster ensemble

Setup `ClusterMerger` using the following commands from the root directory of the repository (require cmake, bison, flex):
```bash
cd ClusterMerger
./setup.sh
./easy_build_and_compile.sh
```

### Post-processing

Setup `constrained-clustering` using the following commands from the root directory of the repository (require cmake, bison, flex)
```bash
cd constrained-clustering
./setup.sh
./easy_build_and_compile.sh
```
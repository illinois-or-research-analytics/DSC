# Dense Subgraph Clustering

## Overview

The repository contains the implementation of density-based community detection methods, including the recommended DSC-Flow-Iter and other methods such as DSC-FISTA(int), DSC-FISTA(int)-Iter, DSC-FISTA-Iter, and DSC-Flow.

The repository also contains the script to run a recommended pipeline, which consists of four stages:
1. Running DSC-Flow-Iter on the input network
2. Running Leiden-Mod, RTRex, IKC(5) on the input network
3. Constructing an unweighted consensus network using the constrained voting strategy at the majority rule consensus level
4. Running Leiden-CPM(0.01) on the obtained network from Stage 3 and post-processing the result with WCC.

The preprint of the conference version of the work (which described a different pipeline) and supplementary materials can be found [here](https://doi.org/10.1007/978-3-032-16719-4_3). If you use our work, you can use the following BibTeX entry to cite.
```
@InProceedings{10.1007/978-3-032-16719-4_3,
    author="Vu-Le, The-Anh and Lamy, Jo{\~a}o Alfredo Cardoso and Alessi, Tom{\'a}s and Chen, Ian and Park, Minhyuk and Harb, Elfarouk and Chacko, George and Warnow, Tandy",
    editor="Cherifi, Hocine and Rocha, Luis M. and Cherifi, Chantal and Ertem, Zeynep",
    title="Dense Subgraph Clustering and a New Cluster Ensemble Method",
    booktitle="Complex Networks {\&} Their Applications XIV",
    year="2026",
    publisher="Springer Nature Switzerland",
    address="Cham",
    pages="29--40",
    isbn="978-3-032-16719-4"
}
```
The extended version of the work with the new recommended pipeline (implemented here) is under review for journal submission.

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
./bin/flow-iter examples/input/bitcoin_alpha.csv examples/output/dsc-flow-iter/bitcoin_alpha/com.csv examples/output/dsc-flow-iter/bitcoin_alpha/density.csv
```

```bash
./bin/fista-int-iter 200 examples/input/bitcoin_alpha.csv examples/output/dsc-fista-int/bitcoin_alpha/com.csv examples/output/dsc-fista-int/bitcoin_alpha/density.csv
```

### Running the recommended pipeline

**Command** We can run the recommended pipeline using the following command:
```bash
./pipeline.sh <edgelist> <output_directory>
```
where
- `<edgelist>` is the path to the input edgelist file (CSV format with header `source,target`)
- `<output_directory>` is the path to the output directory where the results will be saved

For example:
```bash
./pipeline.sh examples/input/bitcoin_alpha.csv examples/output/bitcoin_alpha
./pipeline.sh examples/input/dnc.csv examples/output/dnc
```

**Output**

The pipeline writes individual clustering results under `clusterings/` and consensus results under `merge/<merge-id>/`. The default `<merge-id>` for the recommended constrained voting pipeline is `fmrkc-cvc`.

The main output layout is:
```text
<output_directory>/
  clusterings/
    flow-iter/
      com.csv
      density.csv
    leiden-mod/
      com.csv
    RTRex/
      com.csv
    ikc-5/
      com.csv
  merge/
    fmrkc-cvc/
      merged/
        clustering_list.txt
        edge.csv
      unweighted/
        edge.csv
      final/
        com.csv
      final+wcc/
        com.csv
```

The final community detection result for the default pipeline is available at `<output_directory>/merge/fmrkc-cvc/final+wcc/com.csv`. If the pipeline is run with `--merge-id <merge-id>`, replace `fmrkc-cvc` in the path with the provided merge id.

Pipeline stages also write logs and `done` markers next to their outputs so the pipeline can resume completed stages.

## Installation

See [INSTALL.md](INSTALL.md) for build and dependency setup instructions.

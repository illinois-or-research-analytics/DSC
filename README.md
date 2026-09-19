# Dense Subgraph Clustering

## Overview

The repository contains the implementation of density-based community detection methods, including the recommended DSC-Flow-Iter and other methods such as DSC-FISTA(int), DSC-FISTA(int)-Iter, DSC-FISTA-Iter, and DSC-Flow.

The conference version related to the work can be found [here](https://doi.org/10.1007/978-3-032-16719-4_3). The supplementary material is available [here](https://arxiv.org/pdf/2508.17013) (see pages 13-25). If you use our work, please use the following BibTeX entry to cite.
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

The journal version related to the work, with more focus on the cluster ensemble, can be found [here](https://doi.org/10.1007/s41109-026-00809-z). 

All code in [`src`](src) is a modification made by The-Anh Vu-Le of code provided by El-Farouk Harb in May 2025. Harb’s code (available at [this repo](https://github.com/FaroukY/CorporateEquivalence)) solves the dense subgraph problem using FISTA and Flow (Harb et al., NeurIPS 2025). The modifications allow iterative densest subgraph extraction for Flow. For FISTA, the modifications allow integer rounding, combined with iterative densest subgraph extraction for both integer rounding and fractional peeling.

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

**Examples**

```bash
./bin/flow-iter examples/input/dnc.csv examples/output/flow-iter/dnc/com.csv examples/output/flow-iter/dnc/density.csv
```

```bash
./bin/fista-int-iter 200 examples/input/dnc.csv examples/output/fista-int-iter/dnc/com.csv examples/output/fista-int-iter/dnc/density.csv
```

Generated example outputs for all methods are stored under
`examples/output/<method>/dnc/`.

## Installation

See [INSTALL.md](INSTALL.md).

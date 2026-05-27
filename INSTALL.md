# Installation

This repository uses local DSC binaries, Python wrappers, and external tools stored under `externals/`.

## System Dependencies

Install a C/C++ toolchain and the build tools required by the external projects:

```bash
sudo apt-get install build-essential cmake bison flex
```

On non-Debian systems, install the equivalent packages with the system package manager.

## External Submodules

After cloning the repository, initialize the external dependencies:

```bash
git submodule update --init --recursive
```

The external projects are checked out under:

```text
externals/
  ClusterMerger/
  amazon-RTRExtractor/
  constrained-clustering/
```

## DSC Methods

Build the DSC binaries from the repository root:

```bash
bash build.sh
```

This writes the local DSC executables under `bin/`.

## Python Dependencies

Install the Python dependencies used by the wrappers and pipeline:

```bash
pip install pandas python-igraph leidenalg networkit
```

`python-igraph` and `leidenalg` are required for Leiden runs, `networkit` is required for IKC, and `pandas` is used for CSV processing.

## RTRex

RTRex is built from `externals/amazon-RTRExtractor`. In an unpatched checkout, first edit `externals/amazon-RTRExtractor/RTRex/Escape/Nucleus.h` and replace the two `free(stack)` calls with `delete[] stack`. The stack array is allocated with `new[]`, and RTRex builds with `-Werror`.

Then build RTRex from the repository root:

```bash
cd externals/amazon-RTRExtractor/RTRex/clustering
make clean
cd ..
make clean
make
cd ../../..
```

Install the wrapper-visible binary:

```bash
mkdir -p bin
cp externals/amazon-RTRExtractor/RTRex/clustering/RTRex bin/RTRex
chmod +x bin/RTRex
```

## Cluster Ensemble

Build `ClusterMerger` from the repository root:

```bash
cd externals/ClusterMerger
./setup.sh
./easy_build_and_compile.sh
cd ../..
```

The pipeline uses `externals/ClusterMerger/cluster_merger` directly.

## Post-Processing

Build `constrained-clustering` from the repository root:

```bash
cd externals/constrained-clustering
./setup.sh
./easy_build_and_compile.sh
cd ../..
```

The pipeline uses `externals/constrained-clustering/constrained_clustering` directly.

## Verify The Pipeline

Run the bundled examples from the repository root:

```bash
./pipeline.sh examples/input/bitcoin_alpha.csv examples/output/bitcoin_alpha
./pipeline.sh examples/input/dnc.csv examples/output/dnc
```

The default final outputs are:

```text
examples/output/bitcoin_alpha/merge/fmrkc-cvc/final+wcc/com.csv
examples/output/dnc/merge/fmrkc-cvc/final+wcc/com.csv
```

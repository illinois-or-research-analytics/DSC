# Installation

## System Dependencies

Install a C/C++ toolchain:

```bash
sudo apt-get install build-essential
```

On non-Debian systems, install the equivalent packages with the system package manager.

## DSC Methods

Build the DSC binaries from the repository root:

```bash
bash build.sh all
```

This writes the local DSC executables under `bin/`. To build one method, pass
its name instead, for example:

```bash
bash build.sh flow-iter
```

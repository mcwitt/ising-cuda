# ising-cuda

Learning project using the [Ising model](https://en.wikipedia.org/wiki/Ising_model) as a testing ground to cover the following topics:

- Writing efficient GPU kernel code to handle local interactions in $n$ dimensions
- Wrapping C++/CUDA in Python extension modules using [nanobind](https://github.com/wjakob/nanobind)
- Writing GPU kernels in [JAX/Pallas](https://docs.jax.dev/en/latest/pallas/index.html)

## Project map
- `./` contains simple standalone implementations of Monte Carlo simulations of the Ising model. There are C++ reference and CUDA versions for each of the 2-d and general n-d cases.
  - `./python/` contains a Python project (`ising_mcmc`) that implements versions of the above wrapped in an extension module using nanobind; additionally, this includes a JAX and experimental Pallas backend.
    - `./python/src/ising_mcmc/` contains implementations of the `sweeps` function using various backends (currently CPU, CUDA, JAX, and Pallas)
    - `./python/notebooks/` contains some analysis and benchmarking, e.g. comparing the different backends

## Python quickstart

The Python package is pip-installable from `./python`:

```shell
cd ./python
pip install .
```


## Developing

The Python module can be pip-installed in editable mode

```shell
cd ./python
pip install -e .
```

allowing changes to Python files to be reflected in the installed module without running `pip` again. (Note that changes to the C++ code will require running the above command again; this should only recompile the changes, since [scikit-build-core](https://github.com/scikit-build/scikit-build-core) with CMake supports incremental builds.)

Nix Flakes defining development environments are provided. E.g. in `./python/notebooks`, you can run

```shell
nix develop
```

to enter a development shell containing Python with `ising_mcmc` and plotting packages installed.

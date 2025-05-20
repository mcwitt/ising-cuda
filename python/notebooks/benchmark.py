# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %config InlineBackend.figure_format = "retina"

# %%
import hashlib
import subprocess
from contextlib import contextmanager
from multiprocessing import Lock, synchronize
from pathlib import Path
from time import perf_counter
from typing import Literal, cast

import diskcache
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray
from tqdm.auto import tqdm

from ising_mcmc.cuda import sweeps

# %%
cache = diskcache.Cache(".cache")


# %%
@contextmanager
def lock_context(lock: synchronize.Lock):
    lock.acquire()
    try:
        yield
    finally:
        lock.release()


def compile_(ndim: int, lock: synchronize.Lock) -> Path:
    exec_path = Path.cwd() / "build" / "isingnd"
    exec_path.parent.mkdir(exist_ok=True)

    if not exec_path.exists():
        compile_cmd = [
            "nvcc",
            f"-DD={ndim}",
            "--expt-relaxed-constexpr",
            "-O3",
            "isingnd.cu",
            "-lcudart",
            "-lcurand",
            "-o",
            exec_path,
        ]
        with lock_context(lock):
            subprocess.run(compile_cmd, check=True)

    return exec_path


def get_output_path(
    ndim: int,
    size: int,
    h_ext: NDArray[np.floating],
    n_sweeps: int,
    temperatures: NDArray[np.floating],
    seed: int,
) -> Path:
    hash = hashlib.sha256(
        str((ndim, size, h_ext, n_sweeps, temperatures, seed)).encode()
    ).hexdigest()
    return Path(f"out_{hash[:7]}").with_suffix(".csv")


def compile_and_run(
    ndim: int,
    size: int,
    h_ext: NDArray[np.floating],
    n_sweeps: int,
    temperatures: NDArray[np.floating],
    seed: int,
    lock: synchronize.Lock,
    force_rerun: bool = False,
) -> Path:
    output_path = get_output_path(ndim, size, h_ext, n_sweeps, temperatures, seed)

    if not output_path.exists() or force_rerun:
        exec_path = compile_(ndim, lock)
        with output_path.open("w") as fh:
            subprocess.run(
                [exec_path, str(size), "0", "1", str(n_sweeps), str(seed)],
                input="\n".join(str(t) for t in temperatures),
                text=True,
                stdout=fh,
                check=True,
            )

    return output_path


def run_subprocess(
    ndim: int,
    size: int,
    h_ext: NDArray[np.floating],
    n_sweeps: int,
    temperatures: NDArray[np.float32],
    seed: int,
    lock: synchronize.Lock,
    force_rerun: bool = False,
):
    _ = compile_and_run(
        ndim,
        size,
        h_ext,
        n_sweeps,
        temperatures,
        seed=seed,
        lock=lock,
        force_rerun=force_rerun,
    )


def run_wrapped(
    ndim: int,
    size: int,
    n_sweeps: int,
    temperatures: NDArray[np.float32],
    seed: int,
):
    assert ndim == 2
    rng = np.random.default_rng(seed)
    spin = (
        2 * rng.integers(0, 2, size=(len(temperatures), size, size), dtype=np.int32) - 1
    )
    _ = sweeps(
        spin, np.zeros_like(spin, dtype=np.float32), temperatures, n_sweeps, seed
    )


# %%
type Version = Literal["subprocess", "wrapped"]


@cache.memoize()
def measure_wall_time(
    version: Version,
    ndim: int,
    size: int,
    h_ext: NDArray[np.floating],
    n_sweeps: int,
    temperatures: NDArray[np.floating],
    seed: int,
):
    t0 = perf_counter()
    match version:
        case "subprocess":
            run_subprocess(ndim, size, h_ext, n_sweeps, temperatures, seed, Lock())
        case "wrapped":
            run_wrapped(ndim, size, n_sweeps, temperatures, seed)
    t1 = perf_counter()
    return t1 - t0


# %%
ndims = [2]
temperatures = np.array([2.26, 2.27, 2.28], dtype=np.float32)

params_tuples: list[tuple[Version, int, int, float, int]] = [
    (cast(Version, version), ndim, size, h_ext, seed)
    for version in ["subprocess", "wrapped"]
    for ndim in ndims
    for size in [32, 64, 128, 256, 512, 1024]
    for h_ext in [0.0]
    for seed in [0, 1, 2, 3, 4]
]

# ensure compiled
for ndim in ndims:
    compile_(ndim, Lock())

results = pd.DataFrame.from_records(
    [
        (version, ndim, size, seed, wall_time)
        for version, ndim, size, h_ext, seed in tqdm(params_tuples)
        for wall_time in [
            measure_wall_time(
                version,
                ndim,
                size,
                h_ext * np.ones((size,) * ndim),
                10_000,
                temperatures,
                seed,
            )
        ]
    ],
    columns=[
        "version",
        "ndim",
        "size",
        "seed",
        "wall_time",
    ],
).set_index(["version", "ndim", "size", "seed"])

# %%
sns.lineplot(
    results.reset_index(), x="size", y="wall_time", hue="version", estimator="median"
)

# %%
results.unstack("version")

# %%

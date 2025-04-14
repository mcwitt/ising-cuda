# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
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
from multiprocessing import Lock, Pool, synchronize
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from tqdm.auto import tqdm


# %%
@contextmanager
def lock_context(lock: synchronize.Lock):
    lock.acquire()
    try:
        yield
    finally:
        lock.release()


def compile_cpu(grid_size: int, sweeps_per_sample: int, lock: synchronize.Lock) -> Path:
    exec_path = Path.cwd() / f"ising2d_cpu_{grid_size}"

    with lock_context(lock):
        if not exec_path.exists():
            compile_cmd = [
                "cc",
                f"-DL={grid_size}",
                f"-DSWEEPS_PER_SAMPLE={sweeps_per_sample}",
                "-O3",
                "ising2d.c",
                "-lm",
                "-lgsl",
                "-lgslcblas",
                "-o",
                exec_path,
            ]
            subprocess.run(compile_cmd, check=True)

    return exec_path


def compile_gpu(grid_size: int, sweeps_per_sample: int, lock: synchronize.Lock) -> Path:
    exec_path = Path.cwd() / f"ising_gpu_{grid_size}"

    if not exec_path.exists():
        compile_cmd = [
            "nvcc",
            f"-DL={grid_size}",
            f"-DSWEEPS_PER_SAMPLE={sweeps_per_sample}",
            "-O3",
            "ising2d.cu",
            "-lcudart",
            "-lcurand",
            "-o",
            exec_path,
        ]
        with lock_context(lock):
            subprocess.run(compile_cmd, check=True)

    return exec_path


def compile_(
    impl: Literal["cpu", "gpu"],
    grid_size: int,
    sweeps_per_sample: int,
    lock: synchronize.Lock,
) -> Path:
    match impl:
        case "cpu":
            return compile_cpu(grid_size, sweeps_per_sample, lock)
        case "gpu":
            return compile_gpu(grid_size, sweeps_per_sample, lock)


def get_output_path(
    impl: Literal["cpu", "gpu"],
    grid_size: int,
    sweeps_per_sample: int,
    temperatures: list[float],
    seed: int,
) -> Path:
    hash = hashlib.sha256(
        str((grid_size, sweeps_per_sample, temperatures, seed)).encode()
    ).hexdigest()
    return Path(f"out_{impl}_{hash[:7]}").with_suffix(".csv")


def compile_and_run(
    impl: Literal["cpu", "gpu"],
    grid_size: int,
    sweeps_per_sample: int,
    temperatures: list[float],
    n_samples: int,
    seed: int,
    lock: synchronize.Lock,
) -> Path:
    output_path = get_output_path(
        impl, grid_size, sweeps_per_sample, temperatures, seed
    )

    if not output_path.exists():
        exec_path = compile_(impl, grid_size, sweeps_per_sample, lock)
        with output_path.open("w") as fh:
            subprocess.run(
                [exec_path, str(n_samples), str(seed)],
                input="\n".join(str(t) for t in temperatures),
                text=True,
                stdout=fh,
                check=True,
            )

    return output_path


def read_result(
    impl: Literal["cpu", "gpu"],
    grid_size: int,
    sweeps_per_sample: int,
    temperatures: list[float],
    n_samples: int,
    seed: int,
    lock: synchronize.Lock,
) -> pd.DataFrame:
    output_path = compile_and_run(
        impl, grid_size, sweeps_per_sample, temperatures, n_samples, seed, lock
    )
    df = pd.read_csv(output_path)
    df["impl"] = impl
    df = df.set_index(
        ["impl", "grid_size", "sweeps_per_sample", "seed", "temperature", "isample"]
    )
    return df


# %%
lock = Lock()

# %%
pd.read_csv(compile_and_run("cpu", 16, 1_000, [2.2, 2.3], 1_000, 0, lock))


# %%
pd.read_csv(compile_and_run("gpu", 16, 1_000, [2.2, 2.3], 1_000, 0, lock))


# %%
temperatures: list[float] = np.arange(2.20, 2.31, 0.01).tolist()

argss = [
    (impl, grid_size, sweeps_per_sample, temperatures_, n_samples, seed)
    # for CPU, run each temperature in separate process for parallelism
    for impl, grid_sizes_, temperature_batches in [
        ("cpu", [8, 16, 32, 64], [[t] for t in temperatures]),
        ("gpu", [8, 16, 32, 64, 128], [temperatures]),
    ]
    for grid_size in grid_sizes_
    for temperatures_ in temperature_batches
    for sweeps_per_sample in [1_000]
    for n_samples in [1_000]
    for seed in [0]
]

len(argss)


# %%
def run(args):
    impl, grid_size, sweeps_per_sample, temperature, n_samples, seed = args
    return read_result(
        impl, grid_size, sweeps_per_sample, temperature, n_samples, seed, lock
    )


with Pool() as pool:
    results_iter = pool.imap_unordered(run, argss)
    results = list(tqdm(results_iter, total=len(argss)))

data = pd.concat(results).pipe(
    lambda df: df.assign(
        s=df["magnetization"] / df.index.get_level_values("grid_size") ** 2,
    )
)


# %%
def binderg(s, axis):
    return 1.0 - np.mean(s**4, axis) / (3.0 * np.mean(s**2, axis) ** 2)


def bootstrap(df):
    df = pd.DataFrame(
        {
            "g": stats.bootstrap(
                (df["s"],), binderg, vectorized=True
            ).bootstrap_distribution,
        }
    )
    df.index.name = "bs_sample"
    return df


# %%
bs_samples = (
    data[data.index.get_level_values("isample") >= 500]
    .groupby(
        [
            "impl",
            "grid_size",
            "sweeps_per_sample",
            "seed",
            "temperature",
        ]
    )
    .apply(bootstrap)
)
bs_samples.head()

# %%
sns.lineplot(
    bs_samples.reset_index(),
    x="temperature",
    y="g",
    hue="grid_size",
    errorbar="sd",
    marker=".",
)
plt.axvline(2.269, color="gray", ls="--")

# %%
sns.FacetGrid(bs_samples.reset_index(), col="impl").map_dataframe(
    sns.lineplot,
    x="temperature",
    y="g",
    hue="grid_size",
    errorbar="sd",
    marker=".",
).map(lambda *args, **kwargs: plt.axvline(2.269, color="gray", ls="--")).add_legend()

# %%
sns.lineplot(
    data.reset_index(), x="grid_size", y="elapsed_time_s", hue="impl", marker="o"
).set(xscale="log", yscale="log")

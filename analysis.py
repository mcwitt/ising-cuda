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


def compile_cpu(lock: synchronize.Lock) -> Path:
    exec_path = Path.cwd() / "ising2d_cpu"

    with lock_context(lock):
        if not exec_path.exists():
            compile_cmd = [
                "cc",
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


def compile_gpu(lock: synchronize.Lock) -> Path:
    exec_path = Path.cwd() / "ising2d_gpu"

    if not exec_path.exists():
        compile_cmd = [
            "nvcc",
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
    lock: synchronize.Lock,
) -> Path:
    match impl:
        case "cpu":
            return compile_cpu(lock)
        case "gpu":
            return compile_gpu(lock)


def get_output_path(
    impl: Literal["cpu", "gpu"],
    grid_size: int,
    h_ext: float,
    sweeps_per_sample: int,
    temperatures: list[float],
    n_samples: int,
    seed: int,
) -> Path:
    hash = hashlib.sha256(
        str(
            (grid_size, h_ext, sweeps_per_sample, temperatures, n_samples, seed)
        ).encode()
    ).hexdigest()
    return Path(f"out_{impl}_{hash[:7]}").with_suffix(".csv")


def compile_and_run(
    impl: Literal["cpu", "gpu"],
    grid_size: int,
    h_ext: float,
    sweeps_per_sample: int,
    temperatures: list[float],
    n_samples: int,
    seed: int,
    lock: synchronize.Lock,
) -> Path:
    output_path = get_output_path(
        impl, grid_size, h_ext, sweeps_per_sample, temperatures, n_samples, seed
    )

    if not output_path.exists():
        exec_path = compile_(impl, lock)
        with output_path.open("w") as fh:
            subprocess.run(
                [
                    exec_path,
                    str(grid_size),
                    str(h_ext),
                    str(n_samples),
                    str(sweeps_per_sample),
                    str(seed),
                ],
                input="\n".join(str(t) for t in temperatures),
                text=True,
                stdout=fh,
                check=True,
            )

    return output_path


def read_result(
    impl: Literal["cpu", "gpu"],
    grid_size: int,
    h_ext: float,
    sweeps_per_sample: int,
    temperatures: list[float],
    n_samples: int,
    seed: int,
    lock: synchronize.Lock,
) -> pd.DataFrame:
    output_path = compile_and_run(
        impl, grid_size, h_ext, sweeps_per_sample, temperatures, n_samples, seed, lock
    )
    df = pd.read_csv(output_path)
    df["impl"] = impl
    df = df.set_index(
        ["impl", "D", "L", "sweeps_per_sample", "seed", "temperature", "sample"]
    )
    return df


# %%
lock = Lock()

# %%
pd.read_csv(compile_and_run("cpu", 16, 0.0, 1_000, [2.2, 2.3], 5, 0, lock))


# %%
pd.read_csv(compile_and_run("gpu", 16, 0.0, 1_000, [2.2, 2.3], 5, 0, lock))


# %%
temperatures: list[float] = np.linspace(2.0, 2.5, 16).tolist()

argss = [
    (impl, grid_size, h_ext, sweeps_per_sample, temperatures_, n_samples, seed)
    # for CPU, run each temperature in separate process for parallelism
    for impl, grid_sizes_, temperature_batches in [
        ("cpu", [8, 12, 16, 24, 32, 48, 64], [[t] for t in temperatures]),
        ("gpu", [24, 32, 48, 64, 96, 128], [temperatures]),
    ]
    for grid_size in grid_sizes_
    for h_ext in [0.0]
    for temperatures_ in temperature_batches
    for sweeps_per_sample in [1_000]
    for n_samples in [200]
    for seed in [0]
]

len(argss)


# %%
def run(args):
    impl, grid_size, h_ext, sweeps_per_sample, temperature, n_samples, seed = args
    return read_result(
        impl, grid_size, h_ext, sweeps_per_sample, temperature, n_samples, seed, lock
    )


with Pool() as pool:
    results_iter = pool.imap_unordered(run, argss)
    results = list(tqdm(results_iter, total=len(argss)))

data = pd.concat(results)


# %%
def binderg(s2, s4, axis):
    return 1.0 - s4.mean(axis) / (3.0 * s2.mean(axis) ** 2)


def bootstrap(df):
    df = pd.DataFrame(
        {
            "g": stats.bootstrap(
                (df["<m^2>"], df["<m^4>"]), binderg, paired=True, vectorized=True
            ).bootstrap_distribution,
        }
    )
    df.index.name = "bs_sample"
    return df


# %%
bs_samples = (
    data[data.index.get_level_values("sample") >= 100]
    .groupby(
        [
            "impl",
            "L",
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
    hue="L",
    errorbar="sd",
    marker=".",
)
plt.axvline(2.269, color="gray", ls="--")

# %%
sns.FacetGrid(bs_samples.reset_index(), col="impl").map_dataframe(
    sns.lineplot,
    x="temperature",
    y="g",
    hue="L",
    errorbar="sd",
    marker=".",
).map(lambda *args, **kwargs: plt.axvline(2.269, color="gray", ls="--")).add_legend()

# %%
df = data.pipe(
    lambda df: df.join(
        df.groupby(["impl", "L", "sweeps_per_sample", "seed", "sample"])
        .size()
        .rename("n_temps")
    )
    .reset_index()
    .assign(
        sweep_time_s=lambda df: df["time_s"] / df["sweeps_per_sample"],
        sweep_time_per_temp_s=lambda df: np.where(
            df["impl"] == "gpu", df["sweep_time_s"] / df["n_temps"], df["sweep_time_s"]
        ),
    )
)
sns.lineplot(df, x="L", y="sweep_time_per_temp_s", hue="impl", marker="o").set(
    xscale="log", yscale="log"
)

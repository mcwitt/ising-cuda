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
import logging
import subprocess
from contextlib import contextmanager
from multiprocessing import Lock, Pool, synchronize
from pathlib import Path

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


def compile_(grid_size: int, lock: synchronize.Lock) -> Path:
    exec_path = Path.cwd() / f"ising2d_{grid_size}"

    with lock_context(lock):
        if not exec_path.exists():
            compile_cmd = [
                "cc",
                f"-DL={grid_size}",
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


def get_output_path(
    grid_size: int,
    sweeps_per_sample: int,
    temperatures: list[float],
    seed: int,
) -> Path:
    hash = hashlib.sha256(
        str((grid_size, sweeps_per_sample, temperatures, seed)).encode()
    ).hexdigest()
    return Path(f"out_{hash[:7]}").with_suffix(".csv")


def compile_and_run(
    grid_size: int,
    sweeps_per_sample: int,
    temperatures: list[float],
    n_samples: int,
    seed: int,
    lock: synchronize.Lock,
) -> Path:
    output_path = get_output_path(grid_size, sweeps_per_sample, temperatures, seed)

    if not output_path.exists():
        exec_path = compile_(grid_size, lock)
        try:
            with output_path.open("w") as fh:
                subprocess.run(
                    [exec_path, str(n_samples), str(sweeps_per_sample), str(seed)],
                    input="\n".join(str(t) for t in temperatures),
                    text=True,
                    stdout=fh,
                    check=True,
                )
        except subprocess.CalledProcessError:
            logging.exception

    return output_path


def read_result(
    grid_size: int,
    sweeps_per_sample: int,
    temperatures: list[float],
    n_samples: int,
    seed: int,
    lock: synchronize.Lock,
) -> pd.DataFrame:
    output_path = compile_and_run(
        grid_size, sweeps_per_sample, temperatures, n_samples, seed, lock
    )
    return pd.read_csv(output_path)


# %%
lock = Lock()

# %%
pd.read_csv(compile_and_run(8, 1_000, [2.2, 2.3], 1_000, 0, lock))


# %%
argss = [
    (grid_size, sweeps_per_sample, [temperature], n_samples, seed)
    for grid_size in [8, 12, 16, 24, 32]
    for sweeps_per_sample in [1_000]
    for n_samples in [200]
    for temperature in np.arange(2.0, 2.55, 0.05)
    for seed in [0]
]

len(argss)


# %%
def run(args):
    grid_size, sweeps_per_sample, temperature, n_samples, seed = args
    return read_result(grid_size, sweeps_per_sample, temperature, n_samples, seed, lock)


with Pool() as pool:
    results_iter = pool.imap_unordered(run, argss)
    results = list(tqdm(results_iter, total=len(argss)))

data = pd.concat(results).set_index(
    ["L", "sweeps_per_sample", "seed", "temperature", "sample"]
)


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
sns.lineplot(data.reset_index(), x="L", y="time_s", marker="o").set(
    xscale="log", yscale="log"
)

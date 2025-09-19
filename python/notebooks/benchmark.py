# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %config InlineBackend.figure_format = "retina"

# %%
from time import perf_counter
from typing import Literal, cast

import jax
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray
from tqdm.auto import tqdm

from ising_mcmc.cpu.fm import sweeps as sweeps_cpu
from ising_mcmc.cuda.fm import sweeps as sweeps_cuda
from ising_mcmc.jax.fm import sweeps as sweeps_jax

# %%
type Version = Literal["cpu", "cuda", "jax"]


def measure_wall_time(
    version: Version,
    ndim: int,
    size: int,
    n_sweeps: int,
    temperatures: NDArray[np.float32],
    seed: int,
):
    rng = np.random.default_rng(seed)

    spin = rng.integers(
        0, 2, size=(len(temperatures), *(ndim * (size,))), dtype=np.int32
    )
    spin = 2 * spin - 1

    h_ext = np.zeros_like(spin, np.float32)

    t0 = perf_counter()

    if version == "cpu":
        sweeps_cpu(spin, h_ext, temperatures, n_sweeps, seed)
    elif version == "cuda":
        sweeps_cuda(spin, h_ext, temperatures, n_sweeps, seed)
    elif version == "jax":
        keys_by_temp = jax.random.split(
            jax.random.key(seed), (len(temperatures), n_sweeps)
        )
        result = jax.vmap(sweeps_jax)(keys_by_temp, spin, h_ext, temperatures)
        jax.block_until_ready(result)
    else:
        raise ValueError(f"unknown version: {version}")

    t1 = perf_counter()

    return t1 - t0


# %%
temperatures = np.array([2.26, 2.27, 2.28], dtype=np.float32)

params_tuples: list[tuple[Version, int, int, int]] = [
    (cast(Version, version), ndim, size, seed)
    for version in ["cpu", "cuda", "jax"]
    for ndim, size in [
        (2, 32),
        (2, 48),
        (2, 64),
        (2, 96),
        (2, 128),
        (2, 192),
        (2, 256),
        (3, 8),
        (3, 12),
        (3, 16),
        (3, 24),
        (3, 32),
        (3, 40),
        (4, 6),
        (4, 7),
        (4, 8),
        (4, 10),
        (4, 12),
        (4, 14),
        (4, 16),
    ]
    if not (version == "cpu" and size**ndim > 2**16)
    for seed in range(5)
]


results = pd.DataFrame.from_records(
    [
        (version, ndim, size, seed, wall_time)
        for version, ndim, size, seed in tqdm(params_tuples)
        for wall_time in [
            measure_wall_time(
                version,
                ndim,
                size,
                1_000,
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
sns.FacetGrid(results.reset_index(), col="ndim", sharex=False).map_dataframe(
    sns.lineplot, x="size", y="wall_time", hue="version", estimator="median"
).add_legend()

# %%
results.unstack("version")

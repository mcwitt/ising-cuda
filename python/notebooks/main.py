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
import time
from functools import partial

import jax
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from ising_mcmc.cpu.fm import sweeps as sweeps_cpu
from ising_mcmc.cuda.fm import sweeps as sweeps_cuda
from ising_mcmc.jax.fm import sweeps as sweeps_jax
from ising_mcmc.pallas.fm import sweeps as sweeps_pallas

# %%
rng = np.random.default_rng(0)

# %%
l = 512

# %%
tc = 2.269
temps = tc * np.array([0.9, 1.0, 1.1])
nt = len(temps)

# %%
spin = 2 * rng.binomial(1, 0.5, size=(nt, l, l)) - 1
hext = np.zeros((nt, l, l), dtype=np.float32)

# %%
spin.shape

# %%
_, axs = plt.subplots(1, 3, figsize=(10, 3))
for s_t, ax in zip(spin, axs):
    ax.imshow(s_t)
    ax.set_xticks([])
    ax.set_yticks([])

# %%
n_sweeps = 1_000

# %%
# %%time
spin_, accept_rate, m2avg, m4avg = sweeps_cpu(spin, hext, temps, n_sweeps, 0)

# %%
_, axs = plt.subplots(1, 3, figsize=(10, 3))
for s_t, ax in zip(spin_, axs):
    ax.imshow(s_t)
    ax.set_xticks([])
    ax.set_yticks([])

# %%
# %%timeit
spin_, accept_rate, m2avg, m4avg = sweeps_cuda(spin, hext, temps, 1_000, 0)

# %%
_, axs = plt.subplots(1, 3, figsize=(10, 3))
for s_t, ax in zip(spin_, axs):
    ax.imshow(s_t)
    ax.set_xticks([])
    ax.set_yticks([])

# %%
# %%timeit
keys = jax.random.split(jax.random.key(1), (len(temps), n_sweeps))
sweeps_jax_batched = jax.vmap(sweeps_jax)
spin_, accept_rate, m2avg, m4avg = jax.block_until_ready(
    sweeps_jax_batched(keys, spin, hext, temps)
)

# %%
_, axs = plt.subplots(1, 3, figsize=(10, 3))
for s_t, ax in zip(spin_, axs):
    ax.imshow(s_t)
    ax.set_xticks([])
    ax.set_yticks([])

# %%
# %%timeit
keys = jax.random.split(jax.random.key(1), (len(temps), n_sweeps))
sweeps_pallas_batched = jax.vmap(partial(sweeps_pallas, tile_size=16))
spin_, accept_rate, m2avg, m4avg = jax.block_until_ready(
    sweeps_pallas_batched(keys, spin, hext, temps)
)

# %%
_, axs = plt.subplots(1, 3, figsize=(10, 3))
for s_t, ax in zip(spin_, axs):
    ax.imshow(s_t)
    ax.set_xticks([])
    ax.set_yticks([])

# %%
key = jax.random.key(1)
n_trials = 10


def time_sweeps_pallas_batched(key, tile_size: int, n_iters: int, n_warmup: int = 3):
    sweeps_pallas_batched = jax.vmap(partial(sweeps_pallas, tile_size=tile_size))

    def run(key):
        return sweeps_pallas_batched(key, spin, hext, temps)

    key, subkey = jax.random.split(key)
    keys_by_temp_by_iter = jax.random.split(subkey, (n_warmup, len(temps), n_sweeps))

    for keys_by_temp in keys_by_temp_by_iter:
        _ = jax.block_until_ready(run(keys_by_temp))

    keys_by_temp_by_iter = jax.random.split(key, (n_iters, len(temps), n_sweeps))

    t0 = time.perf_counter()
    for keys in keys_by_temp_by_iter:
        _ = jax.block_until_ready(run(keys))
    t1 = time.perf_counter()

    dt = t1 - t0

    return dt / n_iters


results = [
    {tile_size: time_sweeps_pallas_batched(key, tile_size, 10, 3)}
    for tile_size in tqdm([1, 2, 4, 8, 16, 32, 64])
]

# %%
results

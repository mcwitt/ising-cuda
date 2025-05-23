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
import matplotlib.pyplot as plt

# %%
import numpy as np

from ising_mcmc.cuda.fm import sweeps

# %%
rng = np.random.default_rng(0)

# %%
l = 1024

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
spin_, accept_rate, m2avg, m4avg = sweeps(spin, hext, temps, 10_000, 0)

# %%
_, axs = plt.subplots(1, 3, figsize=(10, 3))
for s_t, ax in zip(spin_, axs):
    ax.imshow(s_t)
    ax.set_xticks([])
    ax.set_yticks([])

from functools import partial

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


@partial(jax.vmap, in_axes=(0, 0, 0, 0, None))
def sweeps(
    key: ArrayLike,
    spin: ArrayLike,
    h_ext: ArrayLike,
    temperature: ArrayLike,
    n_sweeps: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    keys = jax.random.split(key, n_sweeps)

    def body(
        carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        key: jax.Array,
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array], None]:
        spin, n_accepted, m2, m4 = carry

        spin, n_accepted_local = sweep(key, spin, h_ext, temperature)

        n_accepted += n_accepted_local

        m = spin.mean()
        m2 += m**2
        m4 += m**4

        return (spin, n_accepted, m2, m4), None

    init = (jnp.asarray(spin), jnp.array(0), jnp.array(0.0), jnp.array(0.0))

    (spin, n_accepted, m2_sum, m4_sum), _ = jax.lax.scan(body, init, keys)

    return (
        spin,
        n_accepted,
        m2_sum / n_sweeps,
        m4_sum / n_sweeps,
    )


def sweep(
    key: ArrayLike,
    spin: ArrayLike,
    h_ext: ArrayLike,
    temperature: ArrayLike,
) -> tuple[jax.Array, jax.Array]:
    k0, k1 = jax.random.split(key)
    spin, n_accepted_0 = checkerboard_sweep(k0, spin, h_ext, temperature, 0)
    spin, n_accepted_1 = checkerboard_sweep(k1, spin, h_ext, temperature, 1)
    n_accepted_total = n_accepted_0 + n_accepted_1
    return spin, n_accepted_total


def get_neighbor_sum(spin):
    neighbor_sum = 0.0
    for d in range(spin.ndim):
        neighbor_sum += jnp.roll(spin, 1, axis=d)
        neighbor_sum += jnp.roll(spin, -1, axis=d)
    return neighbor_sum


def checkerboard_sweep(
    key: ArrayLike,
    spin: ArrayLike,
    h_ext: ArrayLike,
    temperature: ArrayLike,
    parity: int,
) -> tuple[jax.Array, jax.Array]:
    neighbor_sum = get_neighbor_sum(spin)
    h_total = neighbor_sum + h_ext
    delta_energy = 2.0 * spin * h_total

    accepted = jax.random.bernoulli(key, jnp.exp(-delta_energy / temperature))

    spin = jnp.asarray(spin)
    coords = jnp.indices(spin.shape)
    dist = coords.sum(0)
    mask = (dist % 2) == parity
    accepted &= mask

    spin_next = jnp.where(accepted, -spin, spin)
    n_accepted = jnp.sum(accepted)

    return spin_next, n_accepted

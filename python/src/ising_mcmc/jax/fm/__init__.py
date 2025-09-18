import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


@jax.jit
def sweeps(
    keys: ArrayLike,
    spin: ArrayLike,
    h_ext: ArrayLike,
    temperature: ArrayLike,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    keys = jnp.asarray(keys)
    spin = jnp.asarray(spin)
    h_ext = jnp.asarray(h_ext)
    temperature = jnp.asarray(temperature)

    checkerboard = jnp.indices(spin.shape).sum(0) % 2 == 0

    def body(
        carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        key: jax.Array,
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array], None]:
        spin, n_accepted, m2, m4 = carry

        spin, n_accepted_local = _sweep(key, spin, h_ext, temperature, checkerboard)

        n_accepted += n_accepted_local

        m = spin.mean()
        m2 += m**2
        m4 += m**4

        return (spin, n_accepted, m2, m4), None

    init = (spin, jnp.zeros(()), jnp.zeros(()), jnp.zeros(()))

    (spin, n_accepted, m2_sum, m4_sum), _ = jax.lax.scan(body, init, keys)

    n_sweeps = len(keys)

    return (
        spin,
        n_accepted,
        m2_sum / n_sweeps,
        m4_sum / n_sweeps,
    )


def _sweep(
    key: Array,
    spin: Array,
    h_ext: Array,
    temperature: Array,
    checkerboard: Array,
) -> tuple[jax.Array, jax.Array]:
    noise = jax.random.uniform(key, spin.shape, jnp.float32)

    spin, n_accepted_0 = masked_sweep(spin, h_ext, temperature, noise, checkerboard)
    spin, n_accepted_1 = masked_sweep(spin, h_ext, temperature, noise, ~checkerboard)

    n_accepted_total = n_accepted_0 + n_accepted_1
    return spin, n_accepted_total


def get_neighbor_sum(spin):
    neighbor_sum = 0.0
    for d in range(spin.ndim):
        neighbor_sum += jnp.roll(spin, 1, axis=d)
        neighbor_sum += jnp.roll(spin, -1, axis=d)
    return neighbor_sum


def masked_sweep(
    spin: Array,
    h_ext: Array,
    temperature: Array,
    noise: Array,
    mask: Array,
) -> tuple[jax.Array, jax.Array]:
    neighbor_sum = get_neighbor_sum(spin)
    h_total = neighbor_sum + h_ext
    delta_energy = 2.0 * spin * h_total

    accepted = noise < jnp.exp(-delta_energy / temperature)
    accepted &= mask

    spin_next = jnp.where(accepted, -spin, spin)
    n_accepted = jnp.sum(accepted)

    return spin_next, n_accepted

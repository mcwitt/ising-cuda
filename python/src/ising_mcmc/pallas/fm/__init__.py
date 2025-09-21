from functools import partial

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
from jax import Array


def cdiv(a, b):
    return (a + b - 1) // b


def checkerboard(shape):
    """Equivalent to `np.indices(shape).sum(0) % 2`, but unlike the
    former can be lowered to GPU"""
    idxsum = jnp.zeros((), jnp.int32)
    for x in shape:
        idxsum = jnp.expand_dims(idxsum, -1) + jnp.arange(x)
    return idxsum % 2


def sweep(key, spin, h_ext, temperature, tile_size):
    noise = jax.random.uniform(key, spin.shape)

    def masked_sweep(spin, parity):
        def kernel(
            parity,
            padded_spin_ref,
            hext_ref,
            noise_ref,
            temperature_ref,
            spin_out_ref,
            n_accepted_out_ref,
        ):
            ndim = padded_spin_ref.ndim

            def mkslice(d, offset):
                return pl.ds(pl.program_id(d) * tile_size + offset, tile_size)

            tile = tuple(mkslice(d, 0) for d in range(ndim))
            padded_tile = tuple(mkslice(d, 1) for d in range(ndim))

            spin = padded_spin_ref[padded_tile]
            h_ext = hext_ref[tile]
            noise = noise_ref[tile]

            def get_tile_slice(offset, axis):
                return (
                    *(mkslice(k, 1) for k in range(axis)),
                    mkslice(axis, 1 + offset),
                    *(mkslice(k, 1) for k in range(axis + 1, ndim)),
                )

            neighbor_spins = (
                padded_spin_ref[get_tile_slice(offset, axis)]
                for axis in range(ndim)
                for offset in [-1, 1]
            )

            h_total = sum(neighbor_spins) + h_ext
            delta_energy = 2 * spin * h_total
            temperature = temperature_ref[...]
            accepted = noise < jnp.exp(-delta_energy / temperature)
            site_parity = jnp.logical_xor(
                sum(pl.program_id(d) * tile_size for d in range(ndim)) % 2,
                checkerboard((tile_size,) * ndim),
            )
            mask = site_parity == parity
            accepted_masked = accepted & mask

            spin_out_ref[tile] = jnp.where(accepted_masked, -spin, spin)
            n_accepted_out_ref[...] = jnp.sum(accepted_masked)

        padded_spin = jnp.pad(spin, 1, "wrap")
        grid_size = cdiv(spin.shape[0], tile_size)

        return pl.pallas_call(
            partial(kernel, parity),
            grid=(grid_size,) * spin.ndim,
            out_shape=[
                jax.ShapeDtypeStruct((spin.shape), jnp.int32),
                jax.ShapeDtypeStruct((), jnp.int32),
            ],
        )(padded_spin, h_ext, noise, temperature)

    spin, n_accepted_0 = masked_sweep(spin, 0)
    spin, n_accepted_1 = masked_sweep(spin, 1)

    n_accepted = n_accepted_0 + n_accepted_1

    return spin, n_accepted


def sweeps(keys, spin, h_ext, temperature, tile_size):
    for d, extent in enumerate(spin.shape):
        assert spin.shape[0] == extent, (
            f"require hypercubic lattices, but got extent {spin.shape[0]} "
            f"along axis 0 and extent {extent} along axis {d}"
        )

    def body(
        carry: tuple[Array, Array, Array, Array],
        key: Array,
    ) -> tuple[tuple[Array, Array, Array, Array], None]:
        spin, n_accepted, m2, m4 = carry
        spin, n_accepted_iter = sweep(key, spin, h_ext, temperature, tile_size)

        n_accepted += n_accepted_iter

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

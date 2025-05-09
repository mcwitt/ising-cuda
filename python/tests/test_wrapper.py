import numpy as np
import pytest

from ising_mcmc.cuda import sweeps


def test_sweeps():
    l = 64
    nt = 16

    rng = np.random.default_rng(0)
    spin = 2 * rng.binomial(1, 0.5, size=(nt, l, l)) - 1
    hext = np.zeros((nt, l, l)).astype(np.float32)
    temps = np.linspace(2.0, 3.0, nt, dtype=np.float32)

    result_tuple = sweeps(spin, hext, temps, 100, 0)
    assert isinstance(result_tuple, tuple)
    assert len(result_tuple) == 4
    spin_, accept_rate, m2a, m4a = result_tuple

    assert spin_.shape == (nt, l, l)
    assert set(spin_.ravel().tolist()) == {-1, 1}

    def check_values(values):
        assert isinstance(values, np.ndarray)
        assert values.shape == (len(temps),)
        np.testing.assert_array_less(0.0, values)

    check_values(accept_rate)
    check_values(m2a)
    check_values(m4a)


def test_sweeps_validation():
    with pytest.raises(ValueError, match="spin must have at minimum 2 dimensions"):
        sweeps(np.ones((8,)), np.ones((8,)), np.ones((8,)), 100, 0)

    with pytest.raises(ValueError, match="only hypercubic lattices are supported"):
        sweeps(np.ones((8, 8, 4)), np.ones((8, 8, 4)), np.ones((8,)), 100, 0)

    with pytest.raises(ValueError, match="conflicting sizes 8 and 7 in dimension 0"):
        sweeps(np.ones((8, 8, 8)), np.ones((7, 8, 8)), np.ones((8,)), 100, 0)

    with pytest.raises(ValueError, match="conflicting sizes 8 and 7 in dimension 1"):
        sweeps(np.ones((8, 8, 8)), np.ones((8, 7, 7)), np.ones((8,)), 100, 0)

    with pytest.raises(
        ValueError, match="first dimensions of spin, hext, and temps must match"
    ):
        sweeps(np.ones((8, 8, 8)), np.ones((8, 8, 8)), np.ones((7,)), 100, 0)

    spin = np.ones((8, 8, 8))
    spin[1, 1, 1] = 0
    with pytest.raises(ValueError, match="invalid value in spin: 0"):
        sweeps(spin, np.ones((8, 8, 8)), np.ones((8,)), 100, 0)

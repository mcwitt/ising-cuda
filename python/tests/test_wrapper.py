import numpy as np
import pytest

from ising_mcmc.cuda import sweeps


@pytest.mark.parametrize("nt", [1, 8])
@pytest.mark.parametrize("d,l", [(1, 1), (1, 64), (2, 2), (2, 16), (3, 4), (10, 2)])
def test_sweeps(d, l, nt):
    rng = np.random.default_rng(0)
    shape = (nt,) + (l,) * d
    spin = 2 * rng.binomial(1, 0.5, size=shape) - 1
    hext = np.zeros(shape).astype(np.float32)
    temps = np.linspace(2.20, 2.34, nt, dtype=np.float32)

    result_tuple = sweeps(spin, hext, temps, 300, 0)
    assert isinstance(result_tuple, tuple)
    assert len(result_tuple) == 4
    spin_, accept_rate, m2a, m4a = result_tuple

    assert spin_.shape == shape
    assert set(spin_.ravel().tolist()) <= {-1, 1}

    def check_values(values):
        assert isinstance(values, np.ndarray)
        assert values.shape == (len(temps),)
        assert np.all(0.0 <= values)
        assert np.any(0.0 < values)

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

    with pytest.raises(
        ValueError, match="number of dimensions must be between 1 and 10, but got 11"
    ):
        sweeps(np.ones((2,) * 12), np.ones((2,) * 12), np.ones((2,)), 100, 0)

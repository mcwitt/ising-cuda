#include <cassert>
#include <cmath>
#include <format>
#include <span>
#include <vector>

#include <gsl/gsl_rng.h>

#include "fm.hpp"
#include "hypercube.hpp"

template <unsigned int D>
auto sweep_impl(
    const std::span<const unsigned int> strides,
    const std::span<const double> hext,
    const double temperature,
    gsl_rng *const __restrict__ rng,
    std::span<int> spin) -> unsigned long {

  assert(strides.size() == D + 1);

  const unsigned int l = strides[1];

  unsigned long naccept = 0;

  unsigned int cprev[D], ccurr[D], cnext[D];

  for (unsigned int d = 0; d < D; ++d) {
    cprev[d] = l - 2;
    ccurr[d] = l - 1;
    cnext[d] = 0;
  }

  while (true) {
    unsigned int i = 0;
    for (unsigned int d = 0; d < D; ++d) {
      i += strides[d] * ccurr[d];
    }

    int nbrsum = 0;
    for (unsigned int d = 0; d < D; ++d) {
      unsigned int iprev = 0;
      unsigned int inext = 0;

      // compute indices of forward and reverse neighbors in dimension d
      for (unsigned int dp = 0; dp < D; ++dp) {
        iprev += strides[dp] * ((dp == d) ? cprev[dp] : ccurr[dp]);
        inext += strides[dp] * ((dp == d) ? cnext[dp] : ccurr[dp]);
      }

      nbrsum += spin[iprev] + spin[inext];
    }

    const int s = spin[i];
    const double h = (double)nbrsum + hext[i];
    const double de = 2.0 * (double)s * h;

    if (de <= 0) {
      spin[i] = -s;
      ++naccept;
    } else {
      double prob = exp(-de / temperature);
      double rv = gsl_rng_uniform(rng);
      if (rv < prob) {
        spin[i] *= -1;
        ++naccept;
      }
    }

    // update cprev, ccurr, cnext
    {
      unsigned int d;

      for (d = 0; d < D; ++d) {
        cprev[d] = ccurr[d];
        ccurr[d] = cnext[d];
        if (cnext[d] < l - 1) {
          cnext[d]++;
          break;
        }
        cnext[d] = 0;
      }

      if (d == D)
        break;
    }
  }

  return naccept;
}

using SweepImpl = unsigned long (*)(
    const std::span<const unsigned int>,
    const std::span<const double>,
    const double,
    gsl_rng *const __restrict__,
    std::span<int>);

auto get_hypercube_sweep_impl(const unsigned int d) -> SweepImpl {
  switch (d) {
  case 1:
    return sweep_impl<1>;
  case 2:
    return sweep_impl<2>;
  case 3:
    return sweep_impl<3>;
  case 4:
    return sweep_impl<4>;
  case 5:
    return sweep_impl<5>;
  case 6:
    return sweep_impl<6>;
  case 7:
    return sweep_impl<7>;
  case 8:
    return sweep_impl<8>;
  case 9:
    return sweep_impl<9>;
  case 10:
    return sweep_impl<10>;
  default:
    throw std::invalid_argument(std::format(
        "number of dimensions must be between 1 and 10, but got {}", d));
  }
}

auto sweep(
    const std::span<const unsigned int> strides,
    const std::span<const double> hext,
    const double temperature,
    gsl_rng *const __restrict__ rng,
    std::span<int> spin) -> unsigned long {

  const unsigned int d = strides.size() - 1;
  SweepImpl sweep_ = get_hypercube_sweep_impl(d);
  return sweep_(strides, hext, temperature, rng, spin);
}

auto sum(const std::span<const int> arr) -> int {
  int s = 0;
  for (const int x : arr) {
    s += x;
  }
  return s;
}

auto ising_mcmc::cpu::fm::sweeps(
    const unsigned int d,
    const unsigned int l,
    const std::span<const double> hext,
    const std::span<const double> temps,
    const unsigned int n_sweeps,
    const unsigned long seed,
    std::span<int> spin,
    std::span<double> acceptrate,
    std::span<double> m2,
    std::span<double> m4) -> void {

  const auto strides = compute_strides(d, l);
  const unsigned int n = strides[d];
  std::vector<unsigned int> naccept(temps.size());
  gsl_rng *rng = nullptr;

  gsl_rng_env_setup();
  rng = gsl_rng_alloc(gsl_rng_mt19937);
  gsl_rng_set(rng, seed);

  for (unsigned int itemp = 0; itemp < temps.size(); ++itemp) {
    std::span<int> spin_ = spin.subspan(itemp * n, n);
    naccept[itemp] = 0;
    m2[itemp] = 0.0;
    m4[itemp] = 0.0;

    for (unsigned int isweep = 0; isweep < n_sweeps; ++isweep) {
      const double temperature = temps[itemp];
      naccept[itemp] += sweep(strides, hext, temperature, rng, spin_);
      const int spinsum = sum(spin_);
      double m = (double)spinsum / n;
      m2[itemp] += m * m;
      m4[itemp] += m * m * m * m;
    }

    acceptrate[itemp] = (double)naccept[itemp] / n_sweeps / n;
    m2[itemp] /= (double)n_sweeps;
    m4[itemp] /= (double)n_sweeps;
  }
}

#include <span>

namespace ising_mcmc::cuda::fm {

auto sweeps(
    const unsigned int d,
    const unsigned int l,
    const std::span<const float> hext,
    const std::span<const float> temps,
    const unsigned int n_sweeps,
    const unsigned long seed,
    std::span<int> spin,
    std::span<double> acceptrate,
    std::span<double> m2,
    std::span<double> m4) -> void;
}

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
    std::span<float> acceptrate,
    std::span<float> m2,
    std::span<float> m4) -> void;
}

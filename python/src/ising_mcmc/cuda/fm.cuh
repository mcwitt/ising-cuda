#include <vector>

namespace ising_mcmc::cuda::fm {

auto sweeps(
    const unsigned int d,
    const unsigned int l,
    const unsigned int nt,
    const int *const spin,
    const float *const hext,
    const float *const temps,
    const unsigned int n_sweeps,
    const unsigned long seed)
    -> std::tuple<
        std::vector<int>,    // spin
        std::vector<float>,  // acceptance rate by temperature
        std::vector<float>,  // <m^2>
        std::vector<float>>; // <m^4>
}

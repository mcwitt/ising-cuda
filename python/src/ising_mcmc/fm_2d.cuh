namespace ising_mcmc::fm {

__global__ void k_sweep_2d(
    const unsigned int parity,
    const unsigned int *const __restrict__ d_strides,
    const float *hext,
    const size_t nt,
    const float *temps,
    const float *__restrict__ noise,
    int *const __restrict__ spin,
    unsigned long long *const __restrict__ naccept);

}

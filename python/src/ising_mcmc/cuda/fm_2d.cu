#include "fm_2d.cuh"
#include "kernel_utils.cuh"

__global__ void ising_mcmc::cuda::fm::k_sweep_2d(
    const unsigned int parity,
    const unsigned int *const __restrict__ d_strides,
    const float *hext,
    const size_t nt,
    const float *temps,
    const float *__restrict__ noise,
    int *const __restrict__ spin,
    unsigned long long *const __restrict__ naccept) {

  const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int t = blockIdx.z * blockDim.z + threadIdx.z;

  const unsigned int l = d_strides[1];

  if (t >= nt || i >= l || j >= l)
    return;

  int local_naccept = 0;

  if ((i + j) % 2 == parity) {

    const unsigned int iprev = (i == 0) ? l - 1 : i - 1;
    const unsigned int jprev = (j == 0) ? l - 1 : j - 1;

    const unsigned int inext = (i == l - 1) ? 0 : i + 1;
    const unsigned int jnext = (j == l - 1) ? 0 : j + 1;

    const unsigned int offset = t * l * l;

    const int nbrsum =
        spin[offset + i * l + jprev] + spin[offset + i * l + jnext] +
        spin[offset + iprev * l + j] + spin[offset + inext * l + j];

    const unsigned int idx = offset + i * l + j;
    const float h = static_cast<float>(nbrsum) + hext[idx];
    const int s = spin[idx];
    const float de = static_cast<float>(2 * s) * h;

    if (de <= 0) {
      spin[idx] = -s;
      local_naccept = 1;
    } else {
      const float temp = temps[t];
      const float prob = exp(-de / temp);
      if (noise[idx] < prob) {
        spin[idx] = -s;
        local_naccept = 1;
      }
    }
  }

  k_accum_block_sum(local_naccept, &naccept[t]);
}

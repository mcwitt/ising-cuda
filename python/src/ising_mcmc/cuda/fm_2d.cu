#include "fm_2d.cuh"
#include "kernel_utils.cuh"
#include <cassert>

__global__ void ising_mcmc::cuda::fm::k_sweep_2d(
    const unsigned int parity,
    const unsigned int *const __restrict__ d_strides,
    const float *hext,
    const size_t nt,
    const float *temps,
    const float *__restrict__ noise,
    int *const __restrict__ spin,
    unsigned long long *const __restrict__ naccept) {

  // 1. Compute tile and element indices

  const unsigned int l = d_strides[1];

  const unsigned int jtile = blockIdx.x * blockDim.x; // tile start index
  const unsigned int j = jtile + threadIdx.x;         // element index
  if (j >= l)
    return;

  const unsigned int itile = blockIdx.y * blockDim.y; // tile start index
  const unsigned int i = itile + threadIdx.y;         // element index
  if (i >= l)
    return;

  assert(blockDim.z == 1);
  const unsigned int t = blockIdx.z;
  if (t >= nt)
    return;

  const unsigned int ibatch = t * l * l; // batch start index

  // 2. Load tile into shared memory

  __shared__ int spin_s[TILE_SIZE][TILE_SIZE];
  spin_s[threadIdx.y][threadIdx.x] = spin[ibatch + i * l + j];
  __syncthreads();

  // 3. Compute output element

  int local_naccept = 0;

  if ((i + j) % 2 == parity) {

    const unsigned int iprev = (i == 0) ? l - 1 : i - 1;
    const unsigned int jprev = (j == 0) ? l - 1 : j - 1;

    const unsigned int inext = (i == l - 1) ? 0 : i + 1;
    const unsigned int jnext = (j == l - 1) ? 0 : j + 1;

    auto const get_spin = [=](const unsigned int i,
                              const unsigned int j) -> auto {
      const unsigned int is = i - itile;
      const unsigned int js = j - jtile;

      // NOTE: case where i - itile < 0 handled by wraparound of unsigned int
      return ((is < TILE_SIZE) && (js < TILE_SIZE)) ? spin_s[is][js]
                                                    : spin[ibatch + i * l + j];
    };

    const int nbrsum = get_spin(i, jprev) + get_spin(i, jnext) +
                       get_spin(iprev, j) + get_spin(inext, j);

    const unsigned int idx = ibatch + i * l + j;
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

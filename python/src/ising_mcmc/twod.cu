#include "twod.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda.h>
#include <curand.h>

__global__ void k_sweep(
    const unsigned int parity,
    const unsigned int l,
    const float *hext,
    const unsigned int nt,
    const float *temps,
    const float *__restrict__ noise,
    int *const __restrict__ spin,
    unsigned long long *const __restrict__ naccept) {

  const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int t = blockIdx.z * blockDim.z + threadIdx.z;

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

    const float h = (float)nbrsum + hext[offset + i * l + j];
    const unsigned int idx = offset + i * l + j;
    const int s = spin[idx];
    const float de = 2.0f * (float)s * h;

    if (de <= 0) {
      spin[idx] = -s;
      local_naccept = 1;
    } else {
      const float temp = temps[t];
      const float prob = exp(-static_cast<float>(de) / temp);
      if (noise[idx] < prob) {
        spin[idx] = -s;
        local_naccept = 1;
      }
    }
  }

  k_accum_block_sum(local_naccept, &naccept[t]);
}

__global__ void k_accum_scalar_moments(
    const unsigned int n,
    const unsigned int nt,
    const int *const __restrict__ sum,
    float *const __restrict__ m2sum,
    float *const __restrict__ m4sum) {

  const unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;

  if (t >= nt)
    return;

  float m = static_cast<float>(sum[t]) / static_cast<float>(n);

  atomicAdd(&m2sum[t], m * m);
  atomicAdd(&m4sum[t], m * m * m * m);
}

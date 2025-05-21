#include <cuda.h>
#include <vector>

auto compute_strides(unsigned int d, unsigned int l)
    -> std::vector<unsigned int>;

__global__ void k_sweep(
    const unsigned int parity,
    const unsigned int *const __restrict__ strides,
    const float *const __restrict__ hext,
    const size_t nt,
    const float *const __restrict__ temps,
    const float *const __restrict__ noise,
    int *const __restrict__ spin,
    unsigned long long *const __restrict__ naccept);

using KSweepFunc = void (*)(
    const unsigned int,
    const unsigned int *const __restrict__,
    const float *const __restrict__,
    const size_t,
    const float *const __restrict__,
    const float *const __restrict__,
    int *const __restrict__,
    unsigned long long *const __restrict__);

auto get_k_sweep_func(unsigned int ndim) -> KSweepFunc;

constexpr __host__ __device__ auto ceil_div(unsigned int x, unsigned int y)
    -> unsigned int {
  return (x + y - 1) / y;
}

template <typename T> __device__ void k_accum_block_sum(int &val, T *out) {

  /* Computes the sum of val for all threads in a block and stores the
    result in out. */

  // 1. Compute sum of values in each warp

  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  }

  // At this point, the first thread in each warp ("warp leader") has
  // for its value the sum of the values over the warp.

  // 2. Warp leaders store warp sums in shared memory

  __shared__ int warp_sums[32];

  const unsigned int tid = threadIdx.x + threadIdx.y * blockDim.x;

  if (tid % warpSize == 0) {
    warp_sums[tid / warpSize] = val;
  }

  // 3. Threads in first warp reduce warp sums

  __syncthreads(); // ensure all threads see the final value of warp_sums

  const unsigned int tpb = blockDim.x * blockDim.y;
  const unsigned int nwarps = ceil_div(tpb, warpSize);

  if (tid < warpSize) {
    val = (tid < nwarps) ? warp_sums[tid] : 0;

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    // First warp leader updates out
    if (tid == 0) {
      atomicAdd(out, val);
    }
  }
}

template <typename T>
__global__ void k_accum(
    const unsigned int n,
    const size_t nt,
    const T *const __restrict__ vals,
    T *const __restrict__ out) {

  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int t = blockIdx.y * blockDim.y + threadIdx.y;

  if (t >= nt || i >= n)
    return;

  int local_sum = vals[t * n + i];

  k_accum_block_sum(local_sum, &out[t]);
}

__global__ void k_accum_scalar_moments(
    const unsigned int n,
    const size_t nt,
    const int *const __restrict__ sum,
    float *const __restrict__ m2sum,
    float *const __restrict__ m4sum);

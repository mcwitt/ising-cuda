#pragma once

constexpr __host__ __device__ auto ceil_div(unsigned int x, unsigned int y)
    -> unsigned int {
  return (x + y - 1) / y;
}

template <typename T> __device__ void k_accum_block_sum(int &val, T *out) {

  /* Computes the sum of val for all threads in a block and
    accumulates the result in out. */

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

#include <format>
#include <span>
#include <vector>

#include <curand.h>

#include "fm.cuh"
#include "fm_2d.cuh"
#include "fm_nd.cuh"
#include "hypercube.hpp"

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

using SweepKernel = void (*)(
    const unsigned int,
    const unsigned int *const __restrict__,
    const float *const __restrict__,
    const size_t,
    const float *const __restrict__,
    const float *const __restrict__,
    int *const __restrict__,
    unsigned long long *const __restrict__);

auto get_hypercube_sweep_kernel(const unsigned int d) -> SweepKernel {
  using ising_mcmc::cuda::fm::k_sweep_nd;
  switch (d) {
  case 1:
    return k_sweep_nd<1>;
  case 2:
    return k_sweep_nd<2>;
  case 3:
    return k_sweep_nd<3>;
  case 4:
    return k_sweep_nd<4>;
  case 5:
    return k_sweep_nd<5>;
  case 6:
    return k_sweep_nd<6>;
  case 7:
    return k_sweep_nd<7>;
  case 8:
    return k_sweep_nd<8>;
  case 9:
    return k_sweep_nd<9>;
  case 10:
    return k_sweep_nd<10>;
  default:
    throw std::invalid_argument(
        std::format(
            "number of dimensions must be between 1 and 10, but got {}", d));
  }
}

auto get_sweep_kernel_and_launch_params(
    const unsigned int nt,
    const unsigned int d,
    const unsigned int l,
    const unsigned int n) -> std::tuple<SweepKernel, dim3, dim3> {
  switch (d) {
  case 2:
    using ising_mcmc::cuda::fm::TILE_SIZE;
    return std::make_tuple(
        ising_mcmc::cuda::fm::k_sweep_2d,
        dim3(TILE_SIZE, TILE_SIZE, 1),
        dim3(ceil_div(l, TILE_SIZE), ceil_div(l, TILE_SIZE), nt));
  default:
    return std::make_tuple(
        get_hypercube_sweep_kernel(d), dim3(TPB), dim3(ceil_div(n, TPB), nt));
  }
}

auto ising_mcmc::cuda::fm::sweeps(
    const unsigned int d,
    const unsigned int l,
    const std::span<const float> hext,
    const std::span<const float> temps,
    const unsigned int n_sweeps,
    const unsigned long seed,
    std::span<int> spin,
    std::span<double> acceptrate,
    std::span<double> m2,
    std::span<double> m4) -> void {

  const auto strides = compute_strides(d, l);
  const unsigned int n = strides[d];
  const unsigned int nt = temps.size();

  unsigned int *d_strides;
  int *d_spin;
  float *d_hext;
  float *d_noise;
  float *d_temps;
  unsigned long long *d_naccept;
  int *d_spinsum;

  cudaMalloc(&d_strides, (d + 1) * sizeof(unsigned int));
  cudaMemcpy(
      d_strides,
      strides.data(),
      (d + 1) * sizeof(unsigned int),
      cudaMemcpyHostToDevice);

  cudaMalloc(&d_spin, nt * n * sizeof(int));
  cudaMemcpy(d_spin, spin.data(), nt * n * sizeof(int), cudaMemcpyHostToDevice);

  cudaMalloc(&d_hext, nt * n * sizeof(float));
  cudaMemcpy(
      d_hext, hext.data(), nt * n * sizeof(float), cudaMemcpyHostToDevice);

  cudaMalloc(&d_temps, nt * sizeof(float));
  cudaMemcpy(d_temps, temps.data(), nt * sizeof(float), cudaMemcpyHostToDevice);

  cudaMalloc(&d_noise, nt * n * sizeof(float));
  cudaMalloc(&d_naccept, nt * sizeof(unsigned long long));
  cudaMalloc(&d_spinsum, nt * sizeof(int));

  auto [k_sweep, block_dim, grid_dim] =
      get_sweep_kernel_and_launch_params(nt, d, l, n);

  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, seed);

  cudaMemset(d_naccept, 0, nt * sizeof(unsigned long long));

  for (auto isweep = 0u; isweep < n_sweeps; ++isweep) {
    curandGenerateUniform(gen, d_noise, nt * n);

    // checkerboard updates

    k_sweep<<<grid_dim, block_dim>>>(
        0,
        d_strides,
        d_hext,
        nt,
        d_temps,
        d_noise,
        d_spin,
        d_naccept); // light squares

    k_sweep<<<grid_dim, block_dim>>>(
        1,
        d_strides,
        d_hext,
        nt,
        d_temps,
        d_noise,
        d_spin,
        d_naccept); // dark squares

    // accumulate magnetization

    cudaMemset(d_spinsum, 0, nt * sizeof(int));

    k_accum<<<dim3(ceil_div(n, 256), nt), dim3(256)>>>(
        n, nt, d_spin, d_spinsum);

    std::vector<int> spinsum(nt);
    cudaMemcpy(
        spinsum.data(), d_spinsum, nt * sizeof(int), cudaMemcpyDeviceToHost);
    for (auto t = 0u; t < nt; ++t) {
      double m = spinsum[t] / static_cast<double>(n);
      double m2_ = m * m;
      double m4_ = m2_ * m2_;
      m2[t] += m2_;
      m4[t] += m4_;
    }
  }

  cudaMemcpy(spin.data(), d_spin, nt * n * sizeof(int), cudaMemcpyDeviceToHost);

  std::vector<unsigned long long> naccept(nt);
  cudaMemcpy(
      naccept.data(),
      d_naccept,
      nt * sizeof(unsigned long long),
      cudaMemcpyDeviceToHost);

  for (unsigned int t = 0; t < nt; ++t) {
    acceptrate[t] = static_cast<double>(naccept[t]) / n_sweeps / n;
    m2[t] /= n_sweeps;
    m4[t] /= n_sweeps;
  }

  cudaFree(d_strides);
  cudaFree(d_temps);
  cudaFree(d_spin);
  cudaFree(d_hext);
  cudaFree(d_noise);
  cudaFree(d_naccept);
  cudaFree(d_spinsum);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    throw std::runtime_error(cudaGetErrorString(err));
  }
}

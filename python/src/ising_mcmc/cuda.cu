#include <format>
#include <vector>

#include <curand.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>

#include "hypercube.cuh"
#include "kernel_utils.cuh"
#include "twod.cuh"

namespace nb = nanobind;

auto compute_strides(const unsigned int d, const unsigned int l)
    -> std::vector<unsigned int> {
  std::vector<unsigned int> strides(d + 1);
  strides[0] = 1;
  for (auto i = 1u; i <= d; ++i) {
    strides[i] = strides[i - 1] * l;
  }
  return strides;
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
  switch (d) {
  case 1:
    return k_sweep_hypercube<1>;
  case 2:
    return k_sweep_hypercube<2>;
  case 3:
    return k_sweep_hypercube<3>;
  case 4:
    return k_sweep_hypercube<4>;
  case 5:
    return k_sweep_hypercube<5>;
  case 6:
    return k_sweep_hypercube<6>;
  case 7:
    return k_sweep_hypercube<7>;
  case 8:
    return k_sweep_hypercube<8>;
  case 9:
    return k_sweep_hypercube<9>;
  case 10:
    return k_sweep_hypercube<10>;
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
    return std::make_tuple(
        k_sweep_2d, dim3(16, 16), dim3(ceil_div(l, 16), ceil_div(l, 16), nt));
  default:
    return std::make_tuple(
        get_hypercube_sweep_kernel(d), dim3(256), dim3(ceil_div(n, 256), nt));
  }
}

NB_MODULE(cuda, m) {
  m.def(
      "sweeps",
      [](const nb::ndarray<int, nb::device::cpu> &spin,
         const nb::ndarray<float, nb::device::cpu> &hext,
         const nb::ndarray<float, nb::ndim<1>, nb::device::cpu> &temps,
         const unsigned int n_sweeps,
         const unsigned long seed) {
        if (spin.ndim() < 2) {
          throw std::invalid_argument("spin must have at minimum 2 dimensions");
        }

        const unsigned int nt = spin.shape(0);
        const unsigned int l = spin.shape(1);

        for (auto i = 1u; i < spin.ndim(); ++i) {
          if (spin.shape(i) != l)
            throw std::invalid_argument(
                std::format(
                    "only hypercubic lattices are supported, but got "
                    "conflicting dimensions: spin.shape(1) = {}; "
                    "spin.shape({}) = {}",
                    l,
                    i,
                    spin.shape(i)));
        }

        if (hext.ndim() != spin.ndim()) {
          throw std::invalid_argument(
              std::format(
                  "spin and hext must have same shape, but got conflicting "
                  "numbers of dimensions {} and {}",
                  spin.ndim(),
                  hext.ndim()));
        }

        for (auto i = 0u; i < spin.ndim(); ++i) {
          if (spin.shape(i) != hext.shape(i)) {
            throw std::invalid_argument(
                std::format(
                    "spin and hext must have same shape, but got conflicting "
                    "sizes {} and {} in dimension {}",
                    spin.shape(i),
                    hext.shape(i),
                    i));
          }
        }

        if (temps.size() != nt)
          throw std::invalid_argument(
              std::format(
                  "first dimensions of spin, hext, and temps must match, but "
                  "got {} and {}",
                  spin.shape(0),
                  hext.shape(0),
                  temps.size()));

        for (auto i = 0u; i < spin.size(); ++i) {
          const auto s = spin.data()[i];
          if (s != 1 && s != -1) {
            throw std::invalid_argument(
                std::format("invalid value in spin: {}", s));
          }
        }

        const unsigned int d = spin.ndim() - 1;
        auto strides = compute_strides(d, l);
        const unsigned int n = strides[d];

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
        cudaMemcpy(
            d_spin, spin.data(), nt * n * sizeof(int), cudaMemcpyHostToDevice);

        cudaMalloc(&d_hext, nt * n * sizeof(float));
        cudaMemcpy(
            d_hext,
            hext.data(),
            nt * n * sizeof(float),
            cudaMemcpyHostToDevice);

        cudaMalloc(&d_temps, nt * sizeof(float));
        cudaMemcpy(
            d_temps, temps.data(), nt * sizeof(float), cudaMemcpyHostToDevice);

        cudaMalloc(&d_noise, nt * n * sizeof(float));
        cudaMalloc(&d_naccept, nt * sizeof(unsigned long long));
        cudaMalloc(&d_spinsum, nt * sizeof(int));

        auto [k_sweep, block_dim, grid_dim] =
            get_sweep_kernel_and_launch_params(nt, d, l, n);

        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, seed);

        std::vector<double> m2sum(nt);
        std::vector<double> m4sum(nt);

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
              spinsum.data(),
              d_spinsum,
              nt * sizeof(int),
              cudaMemcpyDeviceToHost);
          for (auto t = 0u; t < nt; ++t) {
            double m = spinsum[t] / static_cast<double>(n);
            double m2 = m * m;
            double m4 = m2 * m2;
            m2sum[t] += m2;
            m4sum[t] += m4;
          }
        }

        std::vector<int> spin_(nt * n);
        cudaMemcpy(
            spin_.data(), d_spin, nt * n * sizeof(int), cudaMemcpyDeviceToHost);

        std::vector<unsigned long long> naccept(nt);
        cudaMemcpy(
            naccept.data(),
            d_naccept,
            nt * sizeof(unsigned long long),
            cudaMemcpyDeviceToHost);

        std::vector<double> acceptrate(nt);
        for (auto t = 0u; t < nt; ++t) {
          acceptrate[t] = static_cast<double>(naccept[t]) / n_sweeps / n;
          m2sum[t] /= n_sweeps;
          m4sum[t] /= n_sweeps;
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

        std::vector<std::size_t> shape(d + 1, l);
        shape[0] = nt;

        return std::make_tuple(
            nb::ndarray<nb::numpy, int, nb::ndim<2>>(
                spin_.data(), d + 1, shape.data(), {})
                .cast(),
            nb::ndarray<nb::numpy, double, nb::ndim<1>>(acceptrate.data(), {nt})
                .cast(),
            nb::ndarray<nb::numpy, double, nb::ndim<1>>(m2sum.data(), {nt})
                .cast(),
            nb::ndarray<nb::numpy, double, nb::ndim<1>>(m4sum.data(), {nt})
                .cast());
      });
}

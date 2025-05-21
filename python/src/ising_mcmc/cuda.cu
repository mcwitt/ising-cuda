#include <curand.h>
#include <format>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>
#include <stdexcept>
#include <vector>

#include "hypercube.cuh"

namespace nb = nanobind;

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
            throw std::invalid_argument(std::format(
                "only hypercubic lattices are supported, but got conflicting "
                "dimensions: spin.shape(1) = {}; spin.shape({}) = {}",
                l,
                i,
                spin.shape(i)));
        }

        if (hext.ndim() != spin.ndim()) {
          throw std::invalid_argument(std::format(
              "spin and hext must have same shape, but got conflicting numbers "
              "of dimensions {} and {}",
              spin.ndim(),
              hext.ndim()));
        }

        for (auto i = 0u; i < spin.ndim(); ++i) {
          if (spin.shape(i) != hext.shape(i)) {
            throw std::invalid_argument(std::format(
                "spin and hext must have same shape, but got conflicting sizes "
                "{} and {} in dimension {}",
                spin.shape(i),
                hext.shape(i),
                i));
          }
        }

        if (temps.size() != nt)
          throw std::invalid_argument(std::format(
              "first dimensions of spin, hext, and temps must match, but got "
              "{} and {}",
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
        float *d_m2sum;
        float *d_m4sum;

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
        cudaMalloc(&d_m2sum, nt * sizeof(float));
        cudaMalloc(&d_m4sum, nt * sizeof(float));

        constexpr dim3 blockDim(32, 1, 1);
        dim3 gridDim(ceil_div(n, blockDim.x), ceil_div(nt, blockDim.z), 1);

        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, seed);

        cudaMemset(d_naccept, 0, nt * sizeof(unsigned long long));

        cudaMemset(d_m2sum, 0, nt * sizeof(float));
        cudaMemset(d_m4sum, 0, nt * sizeof(float));

        for (unsigned int isweep = 0; isweep < n_sweeps; ++isweep) {
          const KSweepFunc k_sweep_d = get_k_sweep_func(d);
          curandGenerateUniform(gen, d_noise, nt * n);

          // checkerboard updates

          static_assert(blockDim.z == 1, "require blockDim.z == 1");

          k_sweep_d<<<gridDim, blockDim>>>(
              0,
              d_strides,
              d_hext,
              nt,
              d_temps,
              d_noise,
              d_spin,
              d_naccept); // light squares

          k_sweep_d<<<gridDim, blockDim>>>(
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

          k_accum<<<dim3(ceil_div(n, 32), nt, 1), dim3(32, 1, 1)>>>(
              n, nt, d_spin, d_spinsum);

          k_accum_scalar_moments<<<ceil_div(nt, 32), 32>>>(
              n, nt, d_spinsum, d_m2sum, d_m4sum);
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

        std::vector<float> m2avg(nt);
        cudaMemcpy(
            m2avg.data(), d_m2sum, nt * sizeof(float), cudaMemcpyDeviceToHost);

        std::vector<float> m4avg(nt);
        cudaMemcpy(
            m4avg.data(), d_m4sum, nt * sizeof(float), cudaMemcpyDeviceToHost);

        std::vector<float> acceptrate(nt);
        for (unsigned int t = 0; t < nt; ++t) {
          acceptrate[t] = (float)naccept[t] / n_sweeps / n;
          m2avg[t] /= n_sweeps;
          m4avg[t] /= n_sweeps;
        }

        cudaFree(d_strides);
        cudaFree(d_temps);
        cudaFree(d_spin);
        cudaFree(d_hext);
        cudaFree(d_noise);
        cudaFree(d_naccept);
        cudaFree(d_spinsum);
        cudaFree(d_m2sum);
        cudaFree(d_m4sum);

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
            nb::ndarray<nb::numpy, float, nb::ndim<1>>(acceptrate.data(), {nt})
                .cast(),
            nb::ndarray<nb::numpy, float, nb::ndim<1>>(m2avg.data(), {nt})
                .cast(),
            nb::ndarray<nb::numpy, float, nb::ndim<1>>(m4avg.data(), {nt})
                .cast());
      });
}

#include <format>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>

#include "cuda/fm.cuh"

namespace nb = nanobind;

NB_MODULE(cuda, m) {
  nb::module_ fm = m.def_submodule("fm", "Ising ferromagnets");
  fm.def(
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
        const unsigned int d = spin.ndim() - 1;
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

        auto [spin_, acceptrate, m2avg, m4avg] = ising_mcmc::cuda::fm::sweeps(
            d, l, nt, spin.data(), hext.data(), temps.data(), n_sweeps, seed);

        std::vector<std::size_t> shape(d + 1, l);
        shape[0] = nt;

        return std::make_tuple(
            nb::ndarray<nb::numpy, int, nb::ndim<2>>(
                spin_.data(), d + 1, shape.data(), {})
                .cast(),
            nb::ndarray<nb::numpy, double, nb::ndim<1>>(acceptrate.data(), {nt})
                .cast(),
            nb::ndarray<nb::numpy, double, nb::ndim<1>>(m2avg.data(), {nt})
                .cast(),
            nb::ndarray<nb::numpy, double, nb::ndim<1>>(m4avg.data(), {nt})
                .cast());
      });
}

#include <format>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>

#include "cpu/fm.hpp"

namespace nb = nanobind;

NB_MODULE(cpu, m) {
  nb::module_ fm = m.def_submodule("fm", "Ising ferromagnets");
  fm.def(
      "sweeps",
      [](const nb::ndarray<int, nb::device::cpu> &spin,
         const nb::ndarray<double, nb::device::cpu> &hext,
         const nb::ndarray<double, nb::ndim<1>, nb::device::cpu> &temps,
         const unsigned int n_sweeps,
         const unsigned long seed) {
        if (spin.ndim() < 2) {
          throw std::invalid_argument("spin must have at minimum 2 dimensions");
        }

        const unsigned int d = spin.ndim() - 1;
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

        if (temps.size() != spin.shape(0))
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

        std::vector<double> hext_(hext.size());
        std::copy(hext.data(), hext.data() + hext.size(), hext_.data());

        std::vector<double> temps_(temps.size());
        std::copy(temps.data(), temps.data() + temps.size(), temps_.data());

        std::vector<int> spin_(spin.size());
        std::copy(spin.data(), spin.data() + spin.size(), spin_.data());

        std::vector<double> acceptrate(temps.size());
        std::vector<double> m2(temps.size());
        std::vector<double> m4(temps.size());

        ising_mcmc::cpu::fm::sweeps(
            d, l, hext_, temps_, n_sweeps, seed, spin_, acceptrate, m2, m4);

        std::vector<std::size_t> shape(d + 1, l);
        const unsigned int nt = temps.size();
        shape[0] = nt;

        return std::make_tuple(
            nb::ndarray<nb::numpy, int, nb::ndim<2>>(
                spin_.data(), d + 1, shape.data(), {})
                .cast(),
            nb::ndarray<nb::numpy, double, nb::ndim<1>>(acceptrate.data(), {nt})
                .cast(),
            nb::ndarray<nb::numpy, double, nb::ndim<1>>(m2.data(), {nt}).cast(),
            nb::ndarray<nb::numpy, double, nb::ndim<1>>(m4.data(), {nt})
                .cast());
      });
}

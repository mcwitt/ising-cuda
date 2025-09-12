#include <vector>

auto compute_strides(const unsigned int d, const unsigned int l)
    -> std::vector<unsigned int> {
  std::vector<unsigned int> strides(d + 1);
  strides[0] = 1;
  for (auto i = 1u; i <= d; ++i) {
    strides[i] = strides[i - 1] * l;
  }
  return strides;
}

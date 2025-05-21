#include <stdexcept>
#include <vector>

#include "hypercube.cuh"

auto compute_strides(unsigned int d, unsigned int l)
    -> std::vector<unsigned int> {
  std::vector<unsigned int> strides(d + 1);
  strides[0] = 1;
  for (auto i = 1u; i <= d; ++i) {
    strides[i] = strides[i - 1] * l;
  }
  return strides;
}

template <std::size_t D>
__global__ void k_sweep(
    const unsigned int parity,
    const unsigned int *const __restrict__ strides,
    const float *const __restrict__ hext,
    const size_t nt,
    const float *const __restrict__ temps,
    const float *const __restrict__ noise,
    int *const __restrict__ spin,
    unsigned long long *const __restrict__ naccept) {

  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int t = blockIdx.y * blockDim.y + threadIdx.y;

  const unsigned int n = strides[D];

  if (t >= nt || i >= n)
    return;

  int local_naccept = 0;

  unsigned int ccurr[D];
  unsigned int rem = i;

  for (int d = D - 1; d >= 0; d--) {
    const unsigned int stride = strides[d];
    ccurr[d] = rem / stride;
    rem %= stride;
  }

  unsigned int dist = 0;
  for (unsigned int k : ccurr) {
    dist += k;
  }

  if (dist % 2 == parity) {
    const unsigned int l = strides[1];

    unsigned int cprev[D];
    unsigned int cnext[D];

    for (int d = 0; d < D; ++d) {
      cprev[d] = (ccurr[d] == 0) ? l - 1 : ccurr[d] - 1;
      cnext[d] = (ccurr[d] == l - 1) ? 0 : ccurr[d] + 1;
    }

    const unsigned int offset = t * n;

    int nbrsum = 0;
    for (int d = 0; d < D; ++d) {
      unsigned int iprev = 0;
      unsigned int inext = 0;

      // compute indices of forward and reverse neighbors in dimension d
      for (int dp = 0; dp < D; ++dp) {
        iprev += strides[dp] * ((dp == d) ? cprev[dp] : ccurr[dp]);
        inext += strides[dp] * ((dp == d) ? cnext[dp] : ccurr[dp]);
      }

      nbrsum += spin[offset + iprev];
      nbrsum += spin[offset + inext];
    }

    const unsigned int idx = offset + i;
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

auto get_k_sweep_func(unsigned int ndim) -> KSweepFunc {
  switch (ndim) {
  case 1:
    return k_sweep<1>;
  case 2:
    return k_sweep<2>;
  case 3:
    return k_sweep<3>;
  case 4:
    return k_sweep<4>;
  case 5:
    return k_sweep<5>;
  case 6:
    return k_sweep<6>;
  case 7:
    return k_sweep<7>;
  case 8:
    return k_sweep<8>;
  case 9:
    return k_sweep<9>;
  case 10:
    return k_sweep<10>;
  default:
    throw std::invalid_argument(
        "number of dimensions must be between 1 and 10");
  }
}

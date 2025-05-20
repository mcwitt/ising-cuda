#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda.h>
#include <curand.h>
#include <vector>

constexpr unsigned int D_ = 2;

constexpr unsigned int WARP_SIZE = 32;

auto compute_strides(unsigned int l) -> std::array<unsigned int, D_ + 1> {
  std::array<unsigned int, D_ + 1> strides{};
  strides[0] = 1;
  for (int i = 1; i <= D_; ++i) {
    strides[i] = strides[i - 1] * l;
  }
  return strides;
}

__constant__ unsigned int c_strides[D_ + 1];

__global__ void k_init_random(
    const unsigned int n,
    const float *const __restrict__ noise,
    int *const __restrict__ spin) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= n)
    return;

  const int p = noise[i] < 0.5;
  spin[i] = 2 * p - 1;
};

constexpr __host__ __device__ auto ceil_div(unsigned int x, unsigned int y)
    -> unsigned int {
  return (x + y - 1) / y;
}

template <typename T> __device__ void k_accum_block_sum(int &val, T *out) {

  /* Computes the sum of val for all threads in a block and stores the
    result in out. */

  // 1. Compute sum of values in each warp

  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  }

  // At this point, the first thread in each warp ("warp leader") has
  // for its value the sum of the values over the warp.

  // 2. Warp leaders store warp sums in shared memory

  __shared__ int warp_sums[WARP_SIZE];

  const unsigned int tid = threadIdx.x + threadIdx.y * blockDim.x;

  if (tid % WARP_SIZE == 0) {
    warp_sums[tid / WARP_SIZE] = val;
  }

  // 3. Threads in first warp reduce warp sums

  __syncthreads(); // ensure all threads see the final value of warp_sums

  const unsigned int tpb = blockDim.x * blockDim.y;
  const unsigned int nwarps = ceil_div(tpb, WARP_SIZE);

  if (tid < WARP_SIZE) {
    val = (tid < nwarps) ? warp_sums[tid] : 0;

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    // First warp leader updates out
    if (tid == 0) {
      atomicAdd(out, val);
    }
  }
}

__global__ void k_sweep(
    const unsigned int parity,
    const float hext,
    const size_t nt,
    const float *temps,
    const float *__restrict__ noise,
    int *const __restrict__ spin,
    unsigned long long *const __restrict__ naccept) {

  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  // reduction of naccept assumes block has consistent temperature
  assert(blockDim.z == 1);
  const unsigned int t = blockIdx.y;

  const unsigned int n = c_strides[D_];

  if (t >= nt || i >= n)
    return;

  int local_naccept = 0;

  unsigned int ccurr[D_];
  unsigned int rem = i;

  for (int d = D_ - 1; d >= 0; d--) {
    const unsigned int stride = c_strides[d];
    ccurr[d] = rem / stride;
    rem %= stride;
  }

  unsigned int dist = 0;
  for (unsigned int k : ccurr) {
    dist += k;
  }

  if (dist % 2 == parity) {
    const unsigned int l = c_strides[1];

    unsigned int cprev[D_];
    unsigned int cnext[D_];

    for (int d = 0; d < D_; ++d) {
      cprev[d] = (ccurr[d] == 0) ? l - 1 : ccurr[d] - 1;
      cnext[d] = (ccurr[d] == l - 1) ? 0 : ccurr[d] + 1;
    }

    const unsigned int offset = t * n;

    int nbrsum = 0;
    for (int d = 0; d < D_; ++d) {
      unsigned int iprev = 0;
      unsigned int inext = 0;

      // compute indices of forward and reverse neighbors in dimension d
      for (int dp = 0; dp < D_; ++dp) {
        iprev += c_strides[dp] * ((dp == d) ? cprev[dp] : ccurr[dp]);
        inext += c_strides[dp] * ((dp == d) ? cnext[dp] : ccurr[dp]);
      }

      nbrsum += spin[offset + iprev];
      nbrsum += spin[offset + inext];
    }

    const float h = static_cast<float>(nbrsum) + hext;
    const unsigned int idx = offset + i;
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

auto parse_float(const char *s) -> float {
  char *endptr = nullptr;
  float r = strtof(s, &endptr);
  if (*endptr != '\0') {
    fprintf(stderr, "Invalid float: %s\n", s);
    exit(1);
  }
  return r;
}

auto parse_long(const char *s) -> long {
  char *endptr = nullptr;
  long r = strtol(s, &endptr, 10);
  if (*endptr != '\0') {
    fprintf(stderr, "Invalid long: %s\n", s);
    exit(1);
  }
  return r;
}

void parse_args(
    int argc,
    char *argv[],
    unsigned int *l,
    float *hext,
    unsigned long *n_samples,
    unsigned long *sweeps_per_sample,
    unsigned long *seed) {
  if (argc != 6) {
    fprintf(
        stderr,
        "Usage: %s L H_EXT N_SAMPLES SWEEPS_PER_SAMPLE SEED\n",
        argv[0]);
    exit(1);
  }
  *l = parse_long(argv[1]);
  *hext = parse_float(argv[2]);
  *n_samples = parse_long(argv[3]);
  *sweeps_per_sample = parse_long(argv[4]);
  *seed = parse_long(argv[5]);
}

auto read_floats() -> std::vector<float> {
  std::vector<float> vals;
  float val;

  while (scanf("%f", &val) == 1) {
    vals.push_back(val);
  }

  return vals;
}

auto main(int argc, char *argv[]) -> int {
  unsigned int l;
  float hext;
  unsigned long n_samples;
  unsigned long sweeps_per_sample;
  unsigned long seed;
  parse_args(argc, argv, &l, &hext, &n_samples, &sweeps_per_sample, &seed);

  auto strides = compute_strides(l);
  const unsigned int n = strides[D_];

  const std::vector<float> temps = read_floats();
  const size_t nt = temps.size();

  int *d_spin;
  float *d_noise;
  float *d_temps;
  unsigned long long *d_naccept;
  int *d_spinsum;

  cudaMalloc(&d_spin, nt * n * sizeof(int));
  cudaMalloc(&d_noise, nt * n * sizeof(float));

  cudaMalloc(&d_temps, nt * sizeof(float));
  cudaMalloc(&d_naccept, nt * sizeof(unsigned long long));
  cudaMalloc(&d_spinsum, nt * sizeof(int));

  cudaMemcpyToSymbol(
      c_strides, strides.data(), (D_ + 1) * sizeof(unsigned int));

  cudaMemcpy(d_temps, temps.data(), nt * sizeof(float), cudaMemcpyHostToDevice);

  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, seed);
  curandGenerateUniform(gen, d_noise, nt * n);

  k_init_random<<<ceil_div(nt * n, 256), 256>>>(nt * n, d_noise, d_spin);

  printf(
      "D,L,h_ext,sweeps_per_sample,seed,temperature,sample,accept_rate,"
      "<m^2>,<m^4>,time_s\n");

  for (int isample = 0; isample < n_samples; ++isample) {
    clock_t start_time = clock();
    std::vector<double> m2sum(nt);
    std::vector<double> m4sum(nt);

    cudaMemset(d_naccept, 0, nt * sizeof(unsigned long long));

    for (int isweep = 0; isweep < sweeps_per_sample; ++isweep) {
      curandGenerateUniform(gen, d_noise, nt * n);

      // checkerboard updates

      constexpr dim3 blockDim(256);
      dim3 gridDim(ceil_div(n, blockDim.x), ceil_div(nt, blockDim.z), 1);

      static_assert(blockDim.z == 1, "require blockDim.z == 1");

      k_sweep<<<gridDim, blockDim>>>(
          0,
          hext,
          nt,
          d_temps,
          d_noise,
          d_spin,
          d_naccept); // light squares

      k_sweep<<<gridDim, blockDim>>>(
          1,
          hext,
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
      for (int t = 0; t < nt; ++t) {
        double m = spinsum[t] / static_cast<double>(n);
        double m2 = m * m;
        double m4 = m2 * m2;
        m2sum[t] += m2;
        m4sum[t] += m4;
      }
    }

    std::vector<unsigned long long> naccept(nt);
    cudaMemcpy(
        naccept.data(),
        d_naccept,
        nt * sizeof(unsigned long long),
        cudaMemcpyDeviceToHost);

    clock_t end_time = clock();

    const double time_s = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    for (int t = 0; t < nt; ++t) {
      const double accept_rate =
          (double)naccept[t] / (double)sweeps_per_sample / n;
      const double m2avg = m2sum[t] / (double)sweeps_per_sample;
      const double m4avg = m4sum[t] / (double)sweeps_per_sample;

      printf(
          "%u,%u,%g,%ld,%ld,%g,%d,%g,%g,%g,%g\n",
          D_,
          l,
          hext,
          sweeps_per_sample,
          seed,
          temps[t],
          isample,
          accept_rate,
          m2avg,
          m4avg,
          time_s);
    }
  }

  cudaFree(d_temps);
  cudaFree(d_spin);
  cudaFree(d_noise);
  cudaFree(d_naccept);
  cudaFree(d_spinsum);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    return 1;
  }

  return 0;
}

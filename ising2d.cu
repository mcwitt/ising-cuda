#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda.h>
#include <curand.h>

constexpr unsigned int D_ = 2;

#ifdef L
constexpr unsigned long L_ = L;
#else
constexpr unsigned long L_ = 64;
#endif

constexpr unsigned long N = L_ * L_;

__global__ void k_init_random(
    const float *const __restrict__ noise, int *const __restrict__ spin) {

  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= N)
    return;

  const int p = noise[i] < 0.5;
  spin[i] = 2 * p - 1;
};

__global__ void k_sweep(
    const unsigned int parity,
    const float hext,
    const float temperature,
    const float *__restrict__ noise,
    int *const __restrict__ spin,
    unsigned long long *const __restrict__ naccept) {

  const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= L_ || j >= L_)
    return;

  if ((i + j) % 2 != parity)
    return;

  const unsigned int iprev = (i == 0) ? L_ - 1 : i - 1;
  const unsigned int jprev = (j == 0) ? L_ - 1 : j - 1;

  const unsigned int inext = (i == L_ - 1) ? 0 : i + 1;
  const unsigned int jnext = (j == L_ - 1) ? 0 : j + 1;

  const int nbrsum = spin[i * L_ + jprev] + spin[i * L_ + jnext] +
                     spin[iprev * L_ + j] + spin[inext * L_ + j];

  const float h = (float)nbrsum + hext;
  const unsigned int idx = i * L + j;
  const float de = 2.0f * (float)spin[idx] * h;

  if (de <= 0) {
    spin[i * L_ + j] *= -1;
    atomicAdd(naccept, 1);
  } else {
    const float prob = exp(-static_cast<float>(de) / temperature);
    if (noise[i * L_ + j] < prob) {
      spin[i * L_ + j] *= -1;
      atomicAdd(naccept, 1);
    }
  }
}

template <typename T>
__global__ void
k_accum(const T *const __restrict__ vals, T *const __restrict__ out) {

  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= N)
    return;

  atomicAdd(out, vals[i]);
}

__global__ void k_accum_scalar_moments(
    const int *const __restrict__ sum,
    float *const __restrict__ m2sum,
    float *const __restrict__ m4sum) {

  float m = static_cast<float>(*sum) / static_cast<float>(N);

  atomicAdd(m2sum, m * m);
  atomicAdd(m4sum, m * m * m * m);
}

constexpr auto ceil_div(const unsigned int x, const unsigned int y)
    -> unsigned int {
  return (x + y - 1) / y;
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
    float *hext,
    long *n_samples,
    long *sweeps_per_sample,
    unsigned long *seed) {
  if (argc != 5) {
    fprintf(
        stderr, "Usage: %s H_EXT N_SAMPLES SWEEPS_PER_SAMPLE SEED\n", argv[0]);
    exit(1);
  }
  *hext = parse_float(argv[1]);
  *n_samples = parse_long(argv[2]);
  *sweeps_per_sample = parse_long(argv[3]);
  *seed = parse_long(argv[4]);
}

auto main(int argc, char *argv[]) -> int {
  float hext;
  long n_samples;
  long sweeps_per_sample;
  unsigned long seed;
  parse_args(argc, argv, &hext, &n_samples, &sweeps_per_sample, &seed);

  int *d_spin;
  float *d_noise;
  unsigned long long *d_naccept;
  int *d_spinsum;
  float *d_m2sum;
  float *d_m4sum;

  cudaMalloc(&d_spin, N * sizeof(int));
  cudaMalloc(&d_noise, N * sizeof(float));
  cudaMalloc(&d_naccept, sizeof(unsigned long long));
  cudaMalloc(&d_spinsum, sizeof(int));
  cudaMalloc(&d_m2sum, sizeof(float));
  cudaMalloc(&d_m4sum, sizeof(float));

  printf("D,L,h_ext,sweeps_per_sample,seed,temperature,sample,accept_rate,"
         "<m^2>,<m^4>,time_s\n");

  float temperature;

  while (scanf("%f", &temperature) == 1) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateUniform(gen, d_noise, N);

    k_init_random<<<ceil_div(N, 32), 32>>>(d_noise, d_spin);

    for (int isample = 0; isample < n_samples; isample++) {

      cudaMemset(d_naccept, 0, sizeof(unsigned long long));
      cudaMemset(d_m2sum, 0, sizeof(float));
      cudaMemset(d_m4sum, 0, sizeof(float));

      clock_t start_time = clock();

      for (int isweep = 0; isweep < sweeps_per_sample; isweep++) {
        curandGenerateUniform(gen, d_noise, N);

        constexpr dim3 blockDim(32, 32);
        dim3 gridDim(ceil_div(L_, blockDim.x), ceil_div(L_, blockDim.y));

        // checkerboard updates
        k_sweep<<<gridDim, blockDim>>>(
            0,
            hext,
            temperature,
            d_noise,
            d_spin,
            d_naccept); // update light squares
        k_sweep<<<gridDim, blockDim>>>(
            1,
            hext,
            temperature,
            d_noise,
            d_spin,
            d_naccept); // update dark squares

        // accumulate magnetization
        cudaMemset(d_spinsum, 0, sizeof(int));
        k_accum<<<ceil_div(N, 32), 32>>>(d_spin, d_spinsum);
        k_accum_scalar_moments<<<1, 1>>>(d_spinsum, d_m2sum, d_m4sum);
      }

      unsigned long long naccept;
      cudaMemcpy(
          &naccept,
          d_naccept,
          sizeof(unsigned long long),
          cudaMemcpyDeviceToHost);

      float m2sum, m4sum;
      cudaMemcpy(&m2sum, d_m2sum, sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(&m4sum, d_m4sum, sizeof(float), cudaMemcpyDeviceToHost);

      clock_t end_time = clock();

      const double accept_rate =
          (double)naccept / (double)sweeps_per_sample / L_ / L_;
      const double m2avg = m2sum / (double)sweeps_per_sample;
      const double m4avg = m4sum / (double)sweeps_per_sample;
      const double time_s = (double)(end_time - start_time) / CLOCKS_PER_SEC;

      printf(
          "%u,%lu,%g,%ld,%ld,%g,%d,%g,%g,%g,%g\n",
          D_,
          L_,
          hext,
          sweeps_per_sample,
          seed,
          temperature,
          isample,
          accept_rate,
          m2avg,
          m4avg,
          time_s);
    }
  }

  cudaFree(d_spin);
  cudaFree(d_noise);
  cudaFree(d_naccept);
  cudaFree(d_spinsum);
  cudaFree(d_m2sum);
  cudaFree(d_m4sum);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    return 1;
  }

  return 0;
}

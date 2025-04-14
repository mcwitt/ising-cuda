#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda.h>
#include <curand.h>

#ifdef L
constexpr unsigned long L_ = L;
#else
constexpr unsigned long L_ = 64;
#endif

#ifdef SWEEPS_PER_SAMPLE
constexpr unsigned long SWEEPS_PER_SAMPLE_ = SWEEPS_PER_SAMPLE;
#else
constexpr unsigned long SWEEPS_PER_SAMPLE_ = 1000;
#endif

__global__ void
k_init_random(const float *__restrict__ noise, int *__restrict__ spin) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= L_ || j >= L_)
    return;

  const int p = noise[i * L_ + j] < 0.5;
  spin[i * L_ + j] = 2 * p - 1;
};

template <unsigned int c>
__global__ void k_sweep(
    float temperature,
    const float *__restrict__ noise,
    int *__restrict__ spin,
    int *__restrict__ naccept) {

  static_assert(c == 0 || c == 1, "c must be 0 or 1");

  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= L_ || j >= L_)
    return;

  if ((i + j) % 2 != c)
    return;

  const unsigned int iprev = (i == 0) ? L_ - 1 : i - 1;
  const unsigned int jprev = (j == 0) ? L_ - 1 : j - 1;

  const unsigned int inext = (i == L_ - 1) ? 0 : i + 1;
  const unsigned int jnext = (j == L_ - 1) ? 0 : j + 1;

  const int nbrsum = spin[i * L_ + jprev] + spin[i * L_ + jnext] +
                     spin[iprev * L_ + j] + spin[inext * L_ + j];
  const int de = 2 * spin[i * L_ + j] * nbrsum;

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

__global__ void k_accum_magnetization(
    const int *__restrict__ spin, int *__restrict__ magnetization) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= L_ || j >= L_)
    return;

  atomicAdd(magnetization, spin[i * L_ + j]);
}

inline auto ceil_div(unsigned int x, unsigned int y) -> unsigned int {
  return (x + y - 1) / y;
}

void parse_args(int argc, char *argv[], long *n_samples, unsigned long *seed) {
  if (argc != 3) {
    fprintf(stderr, "Usage: %s N_SAMPLES SEED\n", argv[0]);
    exit(1);
  }

  char *endptr = nullptr;

  *n_samples = strtol(argv[1], &endptr, 10);
  if (*endptr != '\0') {
    fprintf(stderr, "Invalid integer: %s\n", argv[1]);
    exit(1);
  }

  *seed = strtol(argv[2], &endptr, 10);
  if (*endptr != '\0') {
    fprintf(stderr, "Invalid integer: %s\n", argv[2]);
    exit(1);
  }
}

auto main(int argc, char *argv[]) -> int {
  float temperature;
  long n_samples;
  unsigned long seed;

  int naccept;
  int magnetization;

  int *d_spin;
  float *d_noise;
  int *d_naccept;
  int *d_magnetization;

  curandGenerator_t gen;

  parse_args(argc, argv, &n_samples, &seed);

  cudaMalloc(&d_spin, L_ * L_ * sizeof(int));
  cudaMalloc(&d_noise, L_ * L_ * sizeof(float));
  cudaMalloc(&d_naccept, sizeof(int));
  cudaMalloc(&d_magnetization, sizeof(int));

  dim3 blockDim(16, 16);
  dim3 gridDim(ceil_div(L_, blockDim.x), ceil_div(L_, blockDim.y));

  printf("grid_size,sweeps_per_sample,seed,temperature,isample,naccept,"
         "magnetization,elapsed_time_s\n");

  while (scanf("%f", &temperature) == 1) {

    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);

    curandGenerateUniform(gen, d_noise, L_ * L_);
    k_init_random<<<gridDim, blockDim>>>(d_noise, d_spin);

    for (int isample = 0; isample < n_samples; isample++) {
      cudaMemset(d_naccept, 0, sizeof(int));

      clock_t start_time = clock();

      for (int isweep = 0; isweep < SWEEPS_PER_SAMPLE_; isweep++) {
        curandGenerateUniform(gen, d_noise, L_ * L_);

        // checkerboard updates

        k_sweep<0><<<gridDim, blockDim>>>(
            temperature,
            d_noise,
            d_spin,
            d_naccept); // update light/even squares

        k_sweep<1><<<gridDim, blockDim>>>(
            temperature, d_noise, d_spin, d_naccept); // update dark/odd squares
      }

      clock_t end_time = clock();

      double elapsed_time_s =
          static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC;

      cudaMemcpy(&naccept, d_naccept, sizeof(int), cudaMemcpyDeviceToHost);

      cudaMemset(d_magnetization, 0, sizeof(int));
      k_accum_magnetization<<<gridDim, blockDim>>>(d_spin, d_magnetization);
      cudaMemcpy(
          &magnetization, d_magnetization, sizeof(int), cudaMemcpyDeviceToHost);

      printf(
          "%ld,%ld,%ld,%g,%d,%d,%d,%g\n",
          L_,
          SWEEPS_PER_SAMPLE_,
          seed,
          temperature,
          isample,
          naccept,
          magnetization,
          elapsed_time_s);
    }
  }

  cudaFree(d_spin);
  cudaFree(d_noise);
  cudaFree(d_naccept);
  cudaFree(d_magnetization);

  return 0;
}

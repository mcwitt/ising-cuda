#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#include <gsl/gsl_rng.h>

#ifdef D
constexpr unsigned int D_ = D;
#else
constexpr unsigned int D_ = 64;
#endif

static_assert(D_ >= 1, "Require D >= 1");

#ifdef L
constexpr unsigned int L_ = L;
#else
constexpr unsigned int L_ = 64;
#endif

static_assert(L_ >= 3, "Require L >= 3");

constexpr auto compute_strides() -> std::array<unsigned int, D_ + 1> {
  std::array<unsigned int, D_ + 1> strides{};
  strides[0] = 1;
  for (int i = 1; i <= D_; i++) {
    strides[i] = strides[i - 1] * L_;
  }
  return strides;
}

constexpr auto strides = compute_strides();
constexpr int N = strides[D_];

void init_random(
    gsl_rng *const __restrict__ rng, int *const __restrict__ spin) {
  for (int i = 0; i < N; i++) {
    int p = (int)gsl_rng_uniform_int(rng, 2);
    spin[i] = 2 * p - 1;
  }
};

auto sweep(
    const double hext,
    const double temperature,
    gsl_rng *const __restrict__ rng,
    int *const __restrict__ spin) -> unsigned long {
  unsigned long naccept = 0;

  int cprev[D_], ccurr[D_], cnext[D_];

  for (int d = 0; d < D_; d++) {
    cprev[d] = L_ - 2;
    ccurr[d] = L_ - 1;
    cnext[d] = 0;
  }

  while (true) {
    unsigned int i = 0;
    for (int d = 0; d < D_; d++) {
      i += strides[d] * ccurr[d];
    }

    int nbrsum = 0;
    for (int d = 0; d < D_; d++) {
      unsigned int iprev = 0;
      unsigned int inext = 0;

      // compute indices of forward and reverse neighbors in dimension d
      for (int dp = 0; dp < D_; dp++) {
        iprev += strides[dp] * ((dp == d) ? cprev[dp] : ccurr[dp]);
        inext += strides[dp] * ((dp == d) ? cnext[dp] : ccurr[dp]);
      }

      nbrsum += spin[iprev] + spin[inext];
    }

    const double h = (double)nbrsum + hext;
    const double de = 2.0 * (double)spin[i] * h;

    if (de <= 0) {
      spin[i] *= -1;
      naccept++;
    } else {
      double prob = exp(-de / temperature);
      double rv = gsl_rng_uniform(rng);
      if (rv < prob) {
        spin[i] *= -1;
        naccept++;
      }
    }

    // update cprev, ccurr, cnext
    {
      int d;

      for (d = 0; d < D_; d++) {
        cprev[d] = ccurr[d];
        ccurr[d] = cnext[d];
        if (cnext[d] < L_ - 1) {
          cnext[d]++;
          break;
        }
        cnext[d] = 0;
      }

      if (d == D_)
        break;
    }
  }

  return naccept;
}

auto sum(int n, const int *const __restrict__ arr) -> int {
  int s = 0;
  for (int i = 0; i < n; i++) {
    s += arr[i];
  }
  return s;
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
  int *spin = static_cast<int *>(malloc(N * sizeof(int)));
  float hext = NAN;
  float temperature = NAN;
  long n_samples = 0;
  long sweeps_per_sample = 0;
  unsigned long seed = 0;
  gsl_rng *rng = nullptr;

  parse_args(argc, argv, &hext, &n_samples, &sweeps_per_sample, &seed);

  gsl_rng_env_setup();
  rng = gsl_rng_alloc(gsl_rng_mt19937);
  gsl_rng_set(rng, seed);

  printf("D,L,h_ext,sweeps_per_sample,seed,temperature,sample,accept_rate,"
         "<m^2>,<m^4>,time_s\n");

  while (scanf("%f", &temperature) == 1) {

    init_random(rng, spin);

    for (int isample = 0; isample < n_samples; isample++) {
      unsigned long naccept = 0;

      double m2sum = 0.0;
      double m4sum = 0.0;

      const clock_t start_time = clock();

      for (int isweep = 0; isweep < sweeps_per_sample; isweep++) {
        naccept += sweep(hext, temperature, rng, spin);

        const int spinsum = sum(L_ * L_, spin);
        double m = (double)spinsum / L_ / L_;
        m2sum += m * m;
        m4sum += m * m * m * m;
      }

      const clock_t end_time = clock();

      const double accept_rate =
          (double)naccept / (double)sweeps_per_sample / L_ / L_;
      const double m2avg = m2sum / (double)sweeps_per_sample;
      const double m4avg = m4sum / (double)sweeps_per_sample;
      const double time_s = (double)(end_time - start_time) / CLOCKS_PER_SEC;

      printf(
          "%d,%d,%g,%ld,%ld,%g,%d,%g,%g,%g,%g\n",
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

  return 0;
}

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

auto compute_strides(unsigned int l) -> std::array<unsigned int, D_ + 1> {
  std::array<unsigned int, D_ + 1> strides{};
  strides[0] = 1;
  for (int i = 1; i <= D_; i++) {
    strides[i] = strides[i - 1] * l;
  }
  return strides;
}

void init_random(
    const unsigned int n,
    gsl_rng *const __restrict__ rng,
    int *const __restrict__ spin) {
  for (int i = 0; i < n; i++) {
    int p = (int)gsl_rng_uniform_int(rng, 2);
    spin[i] = 2 * p - 1;
  }
};

auto sweep(
    const std::array<unsigned int, D_ + 1> &strides,
    const double hext,
    const double temperature,
    gsl_rng *const __restrict__ rng,
    int *const __restrict__ spin) -> unsigned long {

  const unsigned int l = strides[1];
  const unsigned int n = strides[D_];

  unsigned long naccept = 0;

  unsigned int cprev[D_], ccurr[D_], cnext[D_];

  for (int d = 0; d < D_; d++) {
    cprev[d] = l - 2;
    ccurr[d] = l - 1;
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
        if (cnext[d] < l - 1) {
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

auto sum(const unsigned int n, const int *const __restrict__ arr) -> int {
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

auto main(int argc, char *argv[]) -> int {
  unsigned int l;
  float hext = NAN;
  float temperature = NAN;
  unsigned long n_samples = 0;
  unsigned long sweeps_per_sample = 0;
  unsigned long seed = 0;
  gsl_rng *rng = nullptr;

  parse_args(argc, argv, &l, &hext, &n_samples, &sweeps_per_sample, &seed);

  auto strides = compute_strides(l);
  const unsigned int n = strides[D_];

  int *spin = static_cast<int *>(malloc(n * sizeof(int)));

  gsl_rng_env_setup();
  rng = gsl_rng_alloc(gsl_rng_mt19937);
  gsl_rng_set(rng, seed);

  printf("D,L,h_ext,sweeps_per_sample,seed,temperature,sample,accept_rate,"
         "<m^2>,<m^4>,time_s\n");

  while (scanf("%f", &temperature) == 1) {

    init_random(n, rng, spin);

    for (int isample = 0; isample < n_samples; isample++) {
      unsigned long naccept = 0;

      double m2sum = 0.0;
      double m4sum = 0.0;

      const clock_t start_time = clock();

      for (int isweep = 0; isweep < sweeps_per_sample; isweep++) {
        naccept += sweep(strides, hext, temperature, rng, spin);

        const int spinsum = sum(n, spin);
        double m = (double)spinsum / n;
        m2sum += m * m;
        m4sum += m * m * m * m;
      }

      const clock_t end_time = clock();

      const double accept_rate =
          (double)naccept / (double)sweeps_per_sample / n;
      const double m2avg = m2sum / (double)sweeps_per_sample;
      const double m4avg = m4sum / (double)sweeps_per_sample;
      const double time_s = (double)(end_time - start_time) / CLOCKS_PER_SEC;

      printf(
          "%d,%d,%g,%ld,%ld,%g,%d,%g,%g,%g,%g\n",
          D_,
          l,
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

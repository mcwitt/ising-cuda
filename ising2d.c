#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <gsl/gsl_rng.h>

#define D 2

_Static_assert(D == 2, "Require D == 2");

#ifndef L
#define L 64
#endif

_Static_assert(L >= 3, "Require L >= 3");

void init_random(gsl_rng *const restrict rng, int *const spin) {
  for (int i = 0; i < L * L; ++i) {
    int p = (int)gsl_rng_uniform_int(rng, 2);
    spin[i] = 2 * p - 1;
  }
};

unsigned int sweep(
    const double hext,
    const double temperature,
    gsl_rng *const restrict rng,
    int *const spin) {
  unsigned int naccept = 0;

  int iprev = L - 2;
  int i = L - 1;

  for (int inext = 0; inext < L; ++inext) {

    int jprev = L - 2;
    int j = L - 1;

    for (int jnext = 0; jnext < L; ++jnext) {
      int nbrsum = spin[i * L + jprev] + spin[i * L + jnext] +
                   spin[iprev * L + j] + spin[inext * L + j];
      const double h = (double)nbrsum + hext;
      const unsigned int idx = i * L + j;
      const double de = 2.0 * (double)spin[idx] * h;

      if (de <= 0) {
        spin[idx] *= -1;
        ++naccept;
      } else {
        double prob = exp(-de / temperature);
        double rv = gsl_rng_uniform(rng);
        if (rv < prob) {
          spin[idx] *= -1;
          ++naccept;
        }
      }

      jprev = j;
      j = jnext;
    }

    iprev = i;
    i = inext;
  }

  return naccept;
}

int sum(const int *const restrict values) {
  int s = 0;
  for (int i = 0; i < L * L; ++i) {
    s += values[i];
  }
  return s;
}

float parse_float(const char *s) {
  char *endptr = NULL;
  float r = strtof(s, &endptr);
  if (*endptr != '\0') {
    fprintf(stderr, "Invalid float: %s\n", s);
    exit(1);
  }
  return r;
}

long parse_long(const char *s) {
  char *endptr = NULL;
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

int main(int argc, char *argv[]) {
  int *spin = (int *)malloc((size_t)L * L * sizeof(int));
  float h = NAN;
  float temperature = NAN;
  long n_samples = 0;
  long sweeps_per_sample = 0;
  unsigned long seed = 0;
  gsl_rng *rng = NULL;

  parse_args(argc, argv, &h, &n_samples, &sweeps_per_sample, &seed);

  gsl_rng_env_setup();
  rng = gsl_rng_alloc(gsl_rng_mt19937);
  gsl_rng_set(rng, seed);

  printf("D,L,h_ext,sweeps_per_sample,seed,temperature,sample,accept_rate,"
         "<m^2>,<m^4>,time_s\n");

  while (scanf("%f", &temperature) == 1) {

    init_random(rng, spin);

    for (int isample = 0; isample < n_samples; ++isample) {
      unsigned int naccept = 0;

      double m2sum = 0.0;
      double m4sum = 0.0;

      const clock_t start_time = clock();

      for (int isweep = 0; isweep < sweeps_per_sample; ++isweep) {
        naccept += sweep(h, temperature, rng, spin);

        const int spinsum = sum(spin);
        double m = (double)spinsum / L / L;
        m2sum += m * m;
        m4sum += m * m * m * m;
      }

      const clock_t end_time = clock();

      const double accept_rate =
          (double)naccept / (double)sweeps_per_sample / L / L;
      const double m2avg = m2sum / (double)sweeps_per_sample;
      const double m4avg = m4sum / (double)sweeps_per_sample;
      const double time_s = (double)(end_time - start_time) / CLOCKS_PER_SEC;

      printf(
          "%d,%d,%g,%ld,%ld,%g,%d,%g,%g,%g,%g\n",
          D,
          L,
          h,
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

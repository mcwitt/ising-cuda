#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <gsl/gsl_rng.h>

#define D 2

_Static_assert(D == 2, "Require D == 2");

void init_random(
    const unsigned int n, gsl_rng *const restrict rng, int *const spin) {
  for (int i = 0; i < n; i++) {
    int p = (int)gsl_rng_uniform_int(rng, 2);
    spin[i] = 2 * p - 1;
  }
};

unsigned int sweep(
    const unsigned int l,
    const double hext,
    const double temperature,
    gsl_rng *const restrict rng,
    int *const spin) {
  unsigned int naccept = 0;

  unsigned int iprev = l - 2;
  unsigned int i = l - 1;

  for (int inext = 0; inext < l; inext++) {

    unsigned int jprev = l - 2;
    unsigned int j = l - 1;

    for (int jnext = 0; jnext < l; jnext++) {
      int nbrsum = spin[i * l + jprev] + spin[i * l + jnext] +
                   spin[iprev * l + j] + spin[inext * l + j];
      const double h = (double)nbrsum + hext;
      const unsigned int idx = i * l + j;
      const double de = 2.0 * (double)spin[idx] * h;

      if (de <= 0) {
        spin[idx] *= -1;
        naccept++;
      } else {
        double prob = exp(-de / temperature);
        double rv = gsl_rng_uniform(rng);
        if (rv < prob) {
          spin[idx] *= -1;
          naccept++;
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

int sum(const unsigned int n, const int *const restrict values) {
  int s = 0;
  for (int i = 0; i < n; i++) {
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

int main(int argc, char *argv[]) {
  unsigned int l;
  float h = NAN;
  float temperature = NAN;
  unsigned long n_samples = 0;
  unsigned long sweeps_per_sample = 0;
  unsigned long seed = 0;
  gsl_rng *rng = NULL;

  parse_args(argc, argv, &l, &h, &n_samples, &sweeps_per_sample, &seed);
  const unsigned int n = l * l;
  int *spin = (int *)malloc((size_t)n * sizeof(int));

  gsl_rng_env_setup();
  rng = gsl_rng_alloc(gsl_rng_mt19937);
  gsl_rng_set(rng, seed);

  printf("D,L,h_ext,sweeps_per_sample,seed,temperature,sample,accept_rate,"
         "<m^2>,<m^4>,time_s\n");

  while (scanf("%f", &temperature) == 1) {

    init_random(n, rng, spin);

    for (int isample = 0; isample < n_samples; isample++) {
      unsigned int naccept = 0;

      double m2sum = 0.0;
      double m4sum = 0.0;

      const clock_t start_time = clock();

      for (int isweep = 0; isweep < sweeps_per_sample; isweep++) {
        naccept += sweep(l, h, temperature, rng, spin);

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
          "%u,%u,%g,%ld,%ld,%g,%d,%g,%g,%g,%g\n",
          D,
          l,
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

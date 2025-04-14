#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <gsl/gsl_rng.h>

#ifndef L
#define L 64
#endif

_Static_assert(L >= 3, "Require L >= 3");

#ifndef SWEEPS_PER_SAMPLE
#define SWEEPS_PER_SAMPLE 1000
#endif

static void init_random(gsl_rng *restrict rng, int spin[][L]) {
  for (int i = 0; i < L; i++) {
    for (int j = 0; j < L; j++) {
      int p = (int)gsl_rng_uniform_int(rng, 2);
      spin[i][j] = 2 * p - 1;
    }
  }
};

static void show(int spin[][L]) {
  for (int i = 0; i < L; i++) {
    for (int j = 0; j < L; j++) {
      printf(spin[i][j] == 1 ? "#" : ".");
    }
    printf("\n");
  }
}

static unsigned long sweep(gsl_rng *restrict rng, int spin[][L], double beta) {
  unsigned long naccept = 0;

  int iprev = L - 2;
  int i = L - 1;

  for (int inext = 0; inext < L; inext++) {

    int jprev = L - 2;
    int j = L - 1;

    for (int jnext = 0; jnext < L; jnext++) {
      int nbrsum =
          spin[i][jprev] + spin[i][jnext] + spin[iprev][j] + spin[inext][j];
      int de = 2 * spin[i][j] * nbrsum;

      if (de <= 0) {
        spin[i][j] *= -1;
        naccept++;
      } else {
        double prob = exp(-beta * de);
        double rv = gsl_rng_uniform(rng);
        if (rv < prob) {
          spin[i][j] *= -1;
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

static int magnetization(int spin[][L]) {
  int magn = 0;
  for (int i = 0; i < L; i++) {
    for (int j = 0; j < L; j++) {
      magn += spin[i][j];
    }
  }
  return magn;
}

void parse_args(int argc, char *argv[], long *n_samples, unsigned long *seed) {
  if (argc != 3) {
    fprintf(stderr, "Usage: %s N_SAMPLES SEED\n", argv[0]);
    exit(1);
  }

  char *endptr = NULL;

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

int main(int argc, char *argv[]) {
  int spin[L][L];
  float temperature = NAN;
  long n_samples = 0;
  unsigned long seed = 0;
  gsl_rng *rng = NULL;

  parse_args(argc, argv, &n_samples, &seed);

  gsl_rng_env_setup();
  rng = gsl_rng_alloc(gsl_rng_mt19937);
  gsl_rng_set(rng, seed);

  printf("grid_size,sweeps_per_sample,seed,temperature,isample,naccept,"
         "magnetization,elapsed_time_s\n");

  while (scanf("%f", &temperature) == 1) {

    init_random(rng, spin);

    for (int isample = 0; isample < n_samples; isample++) {
      unsigned long naccept = 0;
      clock_t start_time = clock();

      for (int isweep = 0; isweep < SWEEPS_PER_SAMPLE; isweep++) {
        naccept += sweep(rng, spin, 1.0 / temperature);
      }

      clock_t end_time = clock();

      double elapsed_time_s = (double)(end_time - start_time) / CLOCKS_PER_SEC;

      int magn = magnetization(spin);

      printf(
          "%d,%d,%ld,%g,%d,%ld,%d,%g\n",
          L,
          SWEEPS_PER_SAMPLE,
          seed,
          temperature,
          isample,
          naccept,
          magn,
          elapsed_time_s);
    }
  }

  return 0;
}

L ?= 16

DEBUG ?= 0

CC := cc

ifeq ($(DEBUG), 0)
	CFLAGS := -O3 -DNDEBUG
else
	CFLAGS := -g -O0
endif

TARGETS := ising2d_cpu

.PHONY: all clean

all: $(TARGETS)

ising2d_cpu: ising2d.c
	$(CC) -DL=$(L) $(CFLAGS) $< -lm -lgsl -lgslcblas -o $@

clean:
	rm -f $(TARGETS)

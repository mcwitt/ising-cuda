L ?= 16

DEBUG ?= 0

CC := cc
NVCC := nvcc

COMMON_FLAGS := -DL=$(L)

ifeq ($(DEBUG), 0)
	COMMON_FLAGS += -O3 -DNDEBUG
else
	COMMON_FLAGS += -g -O0
endif

CFLAGS := $(COMMON_FLAGS)
NVCCFLAGS := $(COMMON_FLAGS)

TARGETS := ising2d_cpu ising2d_gpu

.PHONY: all clean

all: $(TARGETS)

ising2d_cpu: ising2d.c
	$(CC) $(CFLAGS) $< -lm -lgsl -lgslcblas -o $@

ising2d_gpu: ising2d.cu
	$(NVCC) $(NVCCFLAGS) $< -lcudart -lcurand -o $@

clean:
	rm -f $(TARGETS)

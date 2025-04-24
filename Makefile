D ?= 2
L ?= 16

DEBUG ?= 0

CC := cc
NVCC := nvcc

CPU_LIBS := -lm -lgsl -lgslcblas
GPU_LIBS := -lcudart -lcurand

COMMON_FLAGS := -DL=$(L)

ifeq ($(DEBUG), 0)
	COMMON_FLAGS += -O3 -DNDEBUG
else
	COMMON_FLAGS += -g -O0
endif

CFLAGS := $(COMMON_FLAGS)
NVCCFLAGS := $(COMMON_FLAGS)

TARGETS := ising2d_cpu ising2d_gpu isingnd_cpu

.PHONY: all clean

all: $(TARGETS)

ising2d_cpu: ising2d.c
	$(CC) $(CFLAGS) $< $(CPU_LIBS) -o $@

ising2d_gpu: ising2d.cu
	$(NVCC) $(NVCCFLAGS) $< $(GPU_LIBS) -o $@

isingnd_cpu: isingnd.cpp
	$(CC) -DD=$(D) $(CFLAGS) $< $(CPU_LIBS) -o $@

clean:
	rm -f $(TARGETS)

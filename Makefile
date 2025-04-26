D ?= 2
L ?= 16

all: ising2d_cpu ising2d_gpu isingnd_cpu isingnd_gpu

ising2d_cpu: ising2d.c
	cc -DL=$(L) -O3 $< -lm -lgsl -lgslcblas -o $@

ising2d_gpu: ising2d.cu
	nvcc -DL=$(L) -O3 $< -lcudart -lcurand -o $@

isingnd_cpu: isingnd.cpp
	cc -DD=$(D) -DL=$(L) -O3 $< -lm -lgsl -lgslcblas -o $@

isingnd_gpu: isingnd.cu
	nvcc -DD=$(D) -DL=$(L) -O3 --expt-relaxed-constexpr $< -lcudart -lcurand -o $@

clean:
	rm -f ising2d_cpu
	rm -f ising2d_gpu

	rm -f isingnd_cpu
	rm -f isingnd_gpu

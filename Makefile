L ?= 16

all: ising2d_cpu ising2d_gpu

ising2d_cpu: ising2d.c
	cc -DL=$(L) -O3 $< -lm -lgsl -lgslcblas -o $@

ising2d_gpu: ising2d.cu
	nvcc -DL=$(L) -O3 $< -lcudart -lcurand -o $@

clean:
	rm -f ising2d_cpu
	rm -f ising2d_gpu

all: ising2d_cpu ising2d_gpu

ising2d_cpu: ising2d.c
	cc -O3 $< -lm -lgsl -lgslcblas -o $@

ising2d_gpu: ising2d.cu
	nvcc -O3 $< -lcudart -lcurand -o $@

clean:
	rm -f ising2d_cpu
	rm -f ising2d_gpu

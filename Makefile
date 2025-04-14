all: ising2d_cpu

ising2d_cpu: ising2d.c
	cc -O3 $< -lm -lgsl -lgslcblas -o $@

clean:
	rm -f ising2d_cpu

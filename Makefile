nbodyCUDA: nbody.cu
	nvcc -DDUMP nbody.cu -Xcompiler -fopenmp -o nbodyCUDA
nbodyACC: nbody.c
	pgcc -DDUMP nbody.c -fopenmp -o nbodyACC
clean:
	rm -f nbody


nbody: nbody.cu
        nvcc -DDUMP nbody.cu -Xcompiler -fopenmp -o nbody
clean:
        rm -f nbody


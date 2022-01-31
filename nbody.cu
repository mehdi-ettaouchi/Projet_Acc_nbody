#include <stdio.h>
#include <math.h>
#include <stdlib.h> // drand48
#include <omp.h>

//#define DUMP

struct ParticleType { 
  float x, y, z;
  float vx, vy, vz; 
};

__global__ void MoveParticles(const int nParticles, struct ParticleType* particle, const float dt) {
  ParticleType* particle0=particle;
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  // Loop over particles that experience force
  float Fx = 0, Fy = 0, Fz = 0; 
  for (; i < nParticles; i += gridDim.x * blockDim.x) { 

    // Components of the gravity force on particle i
    // Loop over particles that exert force
    for (int j = 0, Fx = Fy = Fz = 0.; j < nParticles; j++) { 
      // No self interaction
      if (i != j) {
          // Avoid singularity and interaction with self
          const float softening = 1e-20;

          // Newton's law of universal gravity
          const float dx = particle0[j].x - particle0[i].x;
          const float dy = particle0[j].y - particle0[i].y;
          const float dz = particle0[j].z - particle0[i].z;
          const float drSquared  = dx*dx + dy*dy + dz*dz + softening;
          const float drPower32  = pow(drSquared, 3.0/2.0);
            
          // Calculate the net force
          Fx += dx / drPower32;  
          Fy += dy / drPower32;  
          Fz += dz / drPower32;
      }

    }

    // Accelerate particles in response to the gravitational force
    particle[i].vx += dt*Fx; 
    particle[i].vy += dt*Fy; 
    particle[i].vz += dt*Fz;
  }

  // Move particles according to their velocities
  // O(N) work, so using a serial loop
  //#pragma acc parallel loop
  i = threadIdx.x + blockIdx.x * blockDim.x;
  for (; i < nParticles; i += gridDim.x * blockDim.x) { 
    particle[i].x  += particle[i].vx*dt;
    particle[i].y  += particle[i].vy*dt;
    particle[i].z  += particle[i].vz*dt;
  }
}

void dump(int iter, int nParticles, struct ParticleType* particle)
{
    char filename[64];
    snprintf(filename, 64, "output_cuda_%d.txt", iter);

    FILE *f;
    f = fopen(filename, "w+");

    int i;
    for (i = 0; i < nParticles; i++)
    {
        fprintf(f, "%e %e %e %e %e %e\n",
                   particle[i].x, particle[i].y, particle[i].z,
		   particle[i].vx, particle[i].vy, particle[i].vz);
    }

    fclose(f);
}

int main(const int argc, const char** argv)
{

  // Problem size and other parameters
  const int nParticles = (argc > 1 ? atoi(argv[1]) : 16384);
  // Duration of test
  const int nSteps = (argc > 2)?atoi(argv[2]):10;
  // Particle propagation time step
  const float dt = 0.0005f;

  struct ParticleType* particle = (struct ParticleType*)malloc(nParticles*sizeof(struct ParticleType));

  // Initialize random number generator and particles
  srand48(0x2020);

  int i;
  for (i = 0; i < nParticles; i++)
  {
     particle[i].x =  2.0*drand48() - 1.0;
     particle[i].y =  2.0*drand48() - 1.0;
     particle[i].z =  2.0*drand48() - 1.0;
     particle[i].vx = 2.0*drand48() - 1.0;
     particle[i].vy = 2.0*drand48() - 1.0;
     particle[i].vz = 2.0*drand48() - 1.0;
  }
  
  // Perform benchmark
  printf("\nPropagating %d particles using 1 thread...\n\n", 
	 nParticles
	 );
  double rate = 0, dRate = 0; // Benchmarking data
  const int skipSteps = 3; // Skip first iteration (warm-up)
  printf("\033[1m%5s %10s %10s %8s\033[0m\n", "Step", "Time, s", "Interact/s", "GFLOP/s"); fflush(stdout);
  for (int step = 1; step <= nSteps; step++) {

    struct ParticleType *d_particles;
    size_t size = nParticles*sizeof(struct ParticleType);
    cudaMalloc(&d_particles, size);
    cudaMemcpy(d_particles, particle, size, cudaMemcpyHostToDevice);

    int threadPerBlocs = 256;
    /* Ceil */
    int blocksPerGrid   = nParticles/threadPerBlocs +((nParticles%threadPerBlocs==0)? 0 : 1) ;

    const double tStart = omp_get_wtime(); // Start timing
    MoveParticles<<< blocksPerGrid, threadPerBlocs >>>(nParticles, d_particles, dt);
    const double tEnd = omp_get_wtime(); // End timing

    cudaMemcpy(particle, d_particles, size, cudaMemcpyDeviceToHost);
    cudaFree(d_particles);

    const float HztoInts   = ((float)nParticles)*((float)(nParticles-1)) ;
    const float HztoGFLOPs = 20.0*1e-9*((float)(nParticles))*((float)(nParticles-1));

    if (step > skipSteps) { // Collect statistics
      rate  += HztoGFLOPs/(tEnd - tStart); 
      dRate += HztoGFLOPs*HztoGFLOPs/((tEnd - tStart)*(tEnd-tStart)); 
    }

    printf("%5d %10.3e %10.3e %8.1f %s\n", 
	   step, (tEnd-tStart), HztoInts/(tEnd-tStart), HztoGFLOPs/(tEnd-tStart), (step<=skipSteps?"*":""));
    fflush(stdout);

#ifdef DUMP
    dump(step, nParticles, particle);
#endif
  }
  rate/=(double)(nSteps-skipSteps); 
  dRate=sqrt(dRate/(double)(nSteps-skipSteps)-rate*rate);
  printf("-----------------------------------------------------\n");
  printf("\033[1m%s %4s \033[42m%10.1f +- %.1f GFLOP/s\033[0m\n",
	 "Average performance:", "", rate, dRate);
  printf("-----------------------------------------------------\n");
  printf("* - warm-up, not included in average\n\n");
  free(particle);
}



#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include "exception.h"
#include "helper_cuda.h"
#include "helper_image.h"
#include "helper_string.h"
#include "helper_timer.h"
#include "helper_functions.h"


#define _USE_MATH_DEFINES

// Print debug messages?
#ifndef DEBUG
#define DEBUG false
#endif

// BLOCKSIZE (number of threads per block)
#ifndef BLOCKSIZE
#define BLOCKSIZE 16
#endif

// NUMTRIALS
#ifndef NUMTRIALS
#define NUMTRIALS 2000
#endif

// NUMBLOCKS
#define NUMBLOCKS NUMTRIALS/BLOCKSIZE

// Ranges for random numbers
const float GMIN = 20.0; // Ground distance
const float GMAX = 30.0;

const float HMIN = 10.0; // Cliff face height
const float HMAX = 40.0;

const float DMIN = 10.0; // Upper-deck distance to castle
const float DMAX = 20.0;

const float VMIN = 30.0; // Initial cannonball velocity
const float VMAX = 50.0;

const float THMIN = 70.0; // Cannon firing angle
const float THMAX = 80.0;

const float GRAVITY = -9.8;
const float TOL = 5.0; // Tolerance in cannonball hitting the castle

// Prototypes
void CudaCheckError();

// Degrees-to-Radians: callable from device
__device__
float
Radians (float d)
{
    return (M_PI/180.f) * d;
}

// The Kernel
__global__
void
MonteCarlo (float *dvs, float *dths, float *dgs, float *dhs, float *dds, int *dhits)
{
    unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;

    // Randomize
    float v = dvs[gid];
    float thr = Radians(dths[gid]);
    float vx = v * cos(thr);
    float vy = v * sin(thr);
    float g = dgs[gid];
    float h = dhs[gid];
    float d = dds[gid];

    // Does the ball reach the cliff?
    float t = ((-vy - (sqrtf(vy*vy - (19.6f*h))))/GRAVITY);
    float x = vx * t;
    if (x <= g)
    {
        // if (DEBUG) fprintf(stderr, "Ball didn't reach the cliff\n");
    }
    // Does the ball hit the vertical cliff face? 
    else
    {
        t = g / vx;
        float y = vy * t + 0.5 * GRAVITY * (t * t);
        if (y <= h)
        {
            // if (DEBUG) fprintf(stderr, "Ball hit the cliff face\n");
        }
        // Does the ball hit the upper deck?
        else
        {
            float a = -4.9;
            float b = vy;
            float c = -h;
            float disc = b * b - 4.f*a*c;

            if (disc < 0.)
            {
                // if (DEBUG) fprintf(stderr, "Ball didn't reach the upper deck\n");
                // exit(1);
            }

            // Successfully hits the ground above the cliffs
            disc = sqrtf(disc);
            float t1 = (-b + disc) / (2.f*a);
            float t2 = (-b - disc) / (2.f*a);

            // We only care about the second intersection
            float tmax = t1;
            if (t2 > t1)
                tmax = t2;
            
            // How far does the ball land horizontally from the edge of the cliff?
            float upperDist = vx * tmax - g;

            // Does it hit the castle? 
            if (fabs(upperDist - d) > TOL)
            {
                // if (DEBUG) fprintf(stderr, "Missed the castle\n");
            }
            else
            {
                // Hits the castle!
                dhits[gid] = 1;
            }
        }
    }

}

float Ranf(float low, float high)
{
    float r = (float) rand();
    float t = r / (float) RAND_MAX;

    return low + t * (high - low);
}

// Defines for labels
#define IN
#define OUT

int 
main (int argc, char* argv[])
{
    int dev = findCudaDevice(argc, (const char**) argv);

    // Define these here so rand() doesn't get into thread timing
    float *hvs = new float [NUMTRIALS];
    float *hths = new float [NUMTRIALS];
    float *hgs = new float [NUMTRIALS];
    float *hhs = new float [NUMTRIALS];
    float *hds = new float [NUMTRIALS];
    int *hhits = new int [NUMTRIALS];

    // Fill the random-value arrays
    for (int n = 0; n < NUMTRIALS; n++)
    {
        hvs[n] = Ranf(VMIN, VMAX);
        hths[n] = Ranf(THMIN, THMAX);
        hgs[n] = Ranf(GMIN, GMAX);
        hhs[n] = Ranf(HMIN, HMAX);
        hds[n] = Ranf(DMIN, DMAX);
    }

    // Allocate device memory
    float *dvs, *dths, *dgs, *dhs, *dds;
    int *dhits;

    cudaMalloc(&dvs, NUMTRIALS*sizeof(float));
    cudaMalloc(&dths, NUMTRIALS*sizeof(float));
    cudaMalloc(&dgs, NUMTRIALS*sizeof(float));
    cudaMalloc(&dhs, NUMTRIALS*sizeof(float));
    cudaMalloc(&dds, NUMTRIALS*sizeof(float));
    cudaMalloc(&dhits, NUMTRIALS*sizeof(float));
    CudaCheckError();

    // Copy host memory to the device
    cudaMemcpy(dvs, hvs, NUMTRIALS*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dths, hths, NUMTRIALS*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dgs, hgs, NUMTRIALS*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dhs, hhs, NUMTRIALS*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dds, hds, NUMTRIALS*sizeof(float), cudaMemcpyHostToDevice);
    CudaCheckError();

    // Setup execution parameters
    dim3 grid(NUMBLOCKS, 1, 1);
    dim3 threads(BLOCKSIZE, 1, 1);

    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    CudaCheckError();

    // Let the GPU go quiet
    cudaDeviceSynchronize();

    // Record the start event
    cudaEventRecord(start, NULL);
    CudaCheckError();

    // Execute the kernel
    MonteCarlo<<< grid, threads >>>(IN dvs, IN dths, IN dgs, IN dhs, IN dds, OUT dhits);

    // Record the stop event
    cudaEventRecord(stop, NULL);
    CudaCheckError();

    // Wait for the stop event to complete
    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);
    CudaCheckError();

    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);
    CudaCheckError();

    // Compute and print performance
    double megaTrialsPerSecond = (double)NUMTRIALS / msecTotal / 1000.;
    fprintf(stderr, "%d blocksize, %d trials; megaTrials/sec = %6.2f\n", BLOCKSIZE, NUMTRIALS, megaTrialsPerSecond);

    // Copy the result from the device to the host
    cudaMemcpy(hhits, dhits, NUMTRIALS*sizeof(int), cudaMemcpyDeviceToHost);
    CudaCheckError();

    // Add up the hhits[] array
    int hits = 0;
    for (int i = 0; i < NUMTRIALS; i++)
    {
        if (hhits[i] != 0)
            hits++;
    }
    
    // Compute and print the probability
    float probability = (float)hits / float(NUMTRIALS);

    fprintf(stderr, "probability: %6.2f%%\n", probability);

    // Clean up host memory
    delete [] hvs;
    delete [] hths;
    delete [] hgs;
    delete [] hhs;
    delete [] hds; 
    delete [] hhits;

    // Clean up device memory
    cudaFree(dvs);
    cudaFree(dths);
    cudaFree(dgs);
    cudaFree(dhs);
    cudaFree(dds);
    cudaFree(dhits);
    CudaCheckError();

    return 0;
}

void CudaCheckError()
{
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
    {
        fprintf(stderr, "CUDA Failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));
    }
}
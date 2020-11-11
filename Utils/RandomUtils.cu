#include "RandomUtils.h"

__global__ void initCurandStates ( curandState * states, unsigned long seed, int maxThreads )
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= maxThreads){
        return;
    }

    curand_init ( seed, index, 0, &states[index] );
}



/*
__global__ void initSobolCurandStates ( curandStateSobol32 * states,int N, unsigned int* directionVectors){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= N){
        return;
    }

    curand_init ( directionVectors, index, &states[index] );
}
*/
#pragma once


#include "../Core/Sampler.h"
#include "../Utils/GpuCommons.h"
#include "../Utils/RandomUtils.h"
#include "../Utils/Array.h"

#include <random>
#include <iostream>

struct HaltonState{
    int dimension;
    unsigned long long index;
};

__device__
inline float runHalton(unsigned int base, unsigned long long i){
    float r = 0;
    float f = 1;
    while(i>0){
        f = f/base;
        r = r + f*(i%base);
        i = i/base;
    }
    return r;
}

#define HALTON_INDEX_SKIP 409ULL

class HaltonSampler: public Sampler{
public:

    int samplesPerPixel;
    int threadsCount;

    GpuArray<HaltonState> states;
    GpuArray<HaltonState> statesCopy; // used for reordering
    GpuArray<unsigned int> primes;
    GpuArray<int> maxDimension;

    __host__
    HaltonSampler(int samplesPerPixel_);

    __host__ 
    HaltonSampler();

    __host__
    HaltonSampler getCopyForKernel();

    virtual void prepare(int threadsCount) override;

    __device__
    virtual void startPixel() override{
        int index = blockIdx.x * blockDim.x + threadIdx.x;

        HaltonState& myState = states.data[index];

        myState.dimension = 0;
        myState.index += HALTON_INDEX_SKIP * (unsigned long)threadsCount;
        *maxDimension.data = 0; // because every thread will reset dimension to 0;
    }

    __device__
    virtual int randInt(int N) override{
        float f = HaltonSampler::rand1();
        int result = f*N; //truncation;
        if (result >= N) result = N - 1;

        return result;
    }

    __device__
	virtual float rand1() override {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        HaltonState& myState = states.data[index];
        unsigned int base = primes.data[myState.dimension];
        myState.dimension += 1;
        return runHalton(base, myState.index);
    };

    __device__
	virtual float2 rand2() override {
        return make_float2(HaltonSampler::rand1(), HaltonSampler::rand1());
    };


    __device__
	virtual float4 rand4() override {
        return make_float4(HaltonSampler::rand1(),HaltonSampler::rand1(),HaltonSampler::rand1(),HaltonSampler::rand1());
    };

    virtual GpuArray<CameraSample> genAllCameraSamples(const CameraObject& camera, FilmObject& film, int bytesNeededPerSample)  override;

    virtual void reorderStates(GpuArray<int>& taskIndices) override;

    virtual void syncDimension() override;

    virtual int bytesNeededPerThread() override {
        return 2*sizeof(HaltonState);
    }
};
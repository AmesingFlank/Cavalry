#pragma once


#include "../Core/Sampler.h"
#include "../Utils/GpuCommons.h"
#include "../Utils/RandomUtils.h"
#include "../Utils/Array.h"

#include <random>
#include <iostream>


__device__
inline float runHalton(unsigned int base, unsigned long long i,unsigned short* perm){
    float r = 0;
    float f = 1;
    while(i>0){
        f = f/base;
        r = r + f*(perm[i%base]);
        i = i/base;
    }
    return r;
}

#define HALTON_INDEX_SKIP 8167ULL //a big prime

class HaltonSampler: public Sampler{
public:

    int samplesPerPixel;
    int threadsCount;

    GpuArray<unsigned int> primes  ;

    GpuArray<unsigned short> permutations ; // array that stores permutation for all bases (primes)
    GpuArray<unsigned int> permutationsStart ;


    __host__
    HaltonSampler(int samplesPerPixel_,bool isCopyForKernel_=false);

    __host__ 
    HaltonSampler();

    __host__
    HaltonSampler getCopyForKernel();

    virtual void prepare(int threadsCount) override;

    __device__
    virtual void startPixel(SamplingState& samplingState, unsigned long long lastIndex) override{
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        samplingState.dimension = 2;
        samplingState.index = lastIndex + 1 + index ;
    }

    __device__
    virtual int randInt(int N, SamplingState& samplingState) override{
        float f = HaltonSampler::rand1(samplingState);
        int result = f*N; //truncation;
        if (result >= N) result = N - 1;

        return result;
    }

    __device__
	virtual float rand1(SamplingState& samplingState) override {
        unsigned int base = primes.data[samplingState.dimension];
        unsigned short* perm = permutations.data + permutationsStart.data[samplingState.dimension];

        samplingState.dimension += 1;
        return runHalton(base, samplingState.index,perm);
    };

    __device__
	virtual float2 rand2(SamplingState& samplingState) override {
        return make_float2(HaltonSampler::rand1(samplingState), HaltonSampler::rand1(samplingState));
    };


    __device__
	virtual float4 rand4(SamplingState& samplingState) override {
        return make_float4(HaltonSampler::rand1(samplingState), HaltonSampler::rand1(samplingState), HaltonSampler::rand1(samplingState), HaltonSampler::rand1(samplingState));
    };

    virtual GpuArray<CameraSample> genAllCameraSamples(const CameraObject& camera, FilmObject& film, int bytesNeededPerSample,int maxSamplesPerRound = -1)  override;

    virtual int bytesNeededPerThread() override {
        return 0;
    }
};
#pragma once


#include "../Core/Sampler.h"
#include "../Utils/GpuCommons.h"
#include "../Utils/RandomUtils.h"
#include "../Utils/Array.h"


#include <random>
#include <iostream>

struct HaltonState{
    int dimension;
    int index;
};

__device__
inline float runHalton(int base, int i){
    float r = 0;
    float f = 1;
    while(i>0){
        f = f/base;
        r = r + f*(i%base);
        i = i/base;
    }
    return r;
}


class HaltonSampler: public Sampler{
public:

    int samplesPerPixel;
    int threadsCount;
    GpuArray<HaltonState> states;
    GpuArray<int> primes;

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
        myState.index += 409;
    }

    __device__
    virtual int randInt(int N) override{

        float f = HaltonSampler::rand1();
        return f*(N-1);

    }

    __device__
	virtual float rand1() override {
        
        int index = blockIdx.x * blockDim.x + threadIdx.x;


        HaltonState& myState = states.data[index];
        int i = myState.index;


        int base = primes.data[myState.dimension];
        myState.dimension += 1;

        return runHalton(base,i);

    };

    __device__
	virtual float2 rand2() override {

        return make_float2(HaltonSampler::rand1(),HaltonSampler::rand1());

    };


    __device__
	virtual float4 rand4() override {

        return make_float4(HaltonSampler::rand1(),HaltonSampler::rand1(),HaltonSampler::rand1(),HaltonSampler::rand1());

    };


    virtual GpuArray<CameraSample> genAllCameraSamples(const CameraObject& camera, FilmObject& film)  override;

};
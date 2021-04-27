#include "SimpleSampler.h"
#include "../Utils/GpuCommons.h"
#include "DecideSampleCount.h"
#include "../Utils/RandomUtils.h"
#include "../Utils/Utils.h"
#include <iostream>


SimpleSampler::SimpleSampler(int samplesPerPixel_, bool isCopyForKernel_ ):states(1024,isCopyForKernel_){
    samplesPerPixel = samplesPerPixel_;
}

SimpleSampler::SimpleSampler() :states(0,true) {

}

SimpleSampler SimpleSampler::getCopyForKernel(){
    SimpleSampler copy(samplesPerPixel,true);
    copy.states = states.getCopyForKernel();
    return copy;
}


__global__
void genNaiveSample(CameraSample* resultPointer, int samplesCount, int width, int height, int samplesPerPixel, SimpleSampler sampler, int pixelIndexStart) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= samplesCount) {
        return;
    }

    SamplingState state;
    state.dimension = 0;
    state.index = index + pixelIndexStart * samplesPerPixel;

    int pixelIndex = pixelIndexStart + index / samplesPerPixel;

    int x = pixelIndex % width;
    int y = pixelIndex / width;

    CameraSample sample{ x , y, state };
    sample.x += sampler.SimpleSampler::rand1(state);
    sample.y += sampler.SimpleSampler::rand1(state);

    resultPointer[index] = sample;
}

GpuArray<CameraSample> SimpleSampler::genAllCameraSamples(const CameraObject& camera, Film& film, int bytesNeededPerSample,int maxSamplesPerRound ) {

    int width = film.width;
    int height = film.height;

    unsigned long long sampleCount = decideSampleCount(film, samplesPerPixel, bytesNeededPerSample);

    unsigned long long pixelsCount = sampleCount / samplesPerPixel;

    prepare(sampleCount);

    GpuArray<CameraSample> result(sampleCount);

    int numBlocks, numThreads;
    setNumBlocksThreads(sampleCount, numBlocks, numThreads);

    genNaiveSample << <numBlocks, numThreads >> > (result.data, sampleCount, width, height, samplesPerPixel, getCopyForKernel(), film.completedPixels);
    CHECK_IF_CUDA_ERROR("gen simple camera samples");

    film.completedPixels += pixelsCount;

    return result;
}
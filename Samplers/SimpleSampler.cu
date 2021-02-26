#include "SimpleSampler.h"
#include "../Utils/GpuCommons.h"
#include "DecideSampleCount.h"
#include "../Utils/RandomUtils.h"
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
void genNaiveSample(CameraSample* resultPointer, int samplesCount, int width, int height,int samplesPerPixel,SimpleSampler sampler){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= samplesCount){
        return;
    }

    int pixelIndex = index / samplesPerPixel;

    int x = pixelIndex % width;
    int y = pixelIndex / width;

    SamplingState unused{ 0,0 };

    CameraSample sample{ (float)x + 0.5 , (float)y +0.5 };
    sample.x += 0.5*sampler.rand1(unused) - 0.25;
    sample.y += 0.5*sampler.rand1(unused) - 0.25;

    resultPointer[index] = sample;
}


GpuArray<CameraSample> SimpleSampler::genAllCameraSamples(const CameraObject& camera, FilmObject& film, int bytesNeededPerSample,int maxSamplesPerRound ) {
    int width = film.getWidth();
    int height = film.getHeight();

    int thisSPP = decideSamplesPerPixel(film,samplesPerPixel,bytesNeededPerSample,maxSamplesPerRound);

    int count = width*height * thisSPP;

    std::cout << "about to alloc cam samples " << thisSPP << std::endl;

    GpuArray<CameraSample> result(count);

    int numThreads = min(count,MAX_THREADS_PER_BLOCK);
    int numBlocks = divUp(count,numThreads);

    genNaiveSample <<<numBlocks,numThreads>>> (result.data,count,width,height,thisSPP,getCopyForKernel());
    CHECK_IF_CUDA_ERROR("gen naive samples");
    return result;
}
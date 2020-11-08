#include "SimpleSamplerGPU.h"
#include "../Utils/GpuCommons.h"

#include "../Utils/RandomUtils.h"
#include <iostream>


SimpleSamplerGPU::SimpleSamplerGPU(int samplesPerPixel_, bool isCopyForKernel_ ):states(1024,isCopyForKernel_),samplesPerPixel(samplesPerPixel_){
    
}

SimpleSamplerGPU::SimpleSamplerGPU() :states(0,true) {

}

SimpleSamplerGPU SimpleSamplerGPU::getCopyForKernel(){
    SimpleSamplerGPU copy(samplesPerPixel,true);
    copy.states = states.getCopyForKernel();
    return copy;
}


__global__
void genNaiveSample(CameraSample* resultPointer, int samplesCount, int width, int height,int samplesPerPixel,SimpleSamplerGPU sampler){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= samplesCount){
        return;
    }

    int pixelIndex = index / samplesPerPixel;

    int x = pixelIndex % width;
    int y = pixelIndex / width;

    CameraSample sample{ (float)x + 0.5 , (float)y +0.5 };
    sample.x += 0.5*sampler.rand1() - 0.25;
    sample.y += 0.5*sampler.rand1() - 0.25;

    resultPointer[index] = sample;
}


GpuArray<CameraSample> SimpleSamplerGPU::genAllCameraSamples(const CameraObject& camera, FilmObject& film) {
    int width = film.getWidth();
    int height = film.getHeight();
    int count = width*height * samplesPerPixel;

    std::cout << "about to alloc cam samples " << samplesPerPixel << std::endl;

    GpuArray<CameraSample> result(count);

    int numThreads = min(count,MAX_THREADS_PER_BLOCK);
    int numBlocks = divUp(count,numThreads);

    genNaiveSample <<<numBlocks,numThreads>>> (result.data,count,width,height,samplesPerPixel,getCopyForKernel());
    CHECK_IF_CUDA_ERROR("gen naive samples");
    return result;
}
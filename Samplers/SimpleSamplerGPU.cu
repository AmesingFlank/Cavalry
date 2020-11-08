#include "SimpleSamplerGPU.h"
#include "../Utils/GpuCommons.h"

#include "../Utils/RandomUtils.h"


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
void genNaiveSample(CameraSample* resultPointer, int samplesCount, int width, int height){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= samplesCount){
        return;
    }
    int x = index % width;
    int y = index / width;
    CameraSample sample{ x +0.5 ,y +0.5 };
    resultPointer[index] = sample;
}


GpuArray<CameraSample> SimpleSamplerGPU::genAllCameraSamples(const CameraObject& camera, FilmObject& film) {
    int width = film.getWidth();
    int height = film.getHeight();
    int count = width*height;
    GpuArray<CameraSample> result(count);

    int numThreads = min(count,MAX_THREADS_PER_BLOCK);
    int numBlocks = divUp(count,numThreads);
    genNaiveSample <<<numBlocks,numThreads>>> (result.data,count,width,height);
    CHECK_CUDA_ERROR("gen naive samples");
    return result;
}
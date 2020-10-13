#include "NaiveCameraSampler.h"
#include "../Utils/GpuCommons.h"

std::vector<CameraSample> NaiveCameraSampler::genAllSamplesCPU(const CameraObject& camera, FilmObject& film) {
    int width = film.getWidth();
    int height = film.getHeight();
    std::vector<CameraSample> result;
    for (float x = 0; x < width; x += 1) {
        for (float y = 0; y < height; y += 1) {
            CameraSample sample{ x / (float)(width - 1),y / (float)(height - 1) };
            result.push_back(sample);
        }
    }
    return result;
}



__global__
void genNaiveSample(CameraSample* resultPointer, int samplesCount, int width, int height){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= samplesCount){
        return;
    }
    int x = index % width;
    int y = index / width;
    CameraSample sample{ x / (float)(width - 1),y / (float)(height - 1) };
    resultPointer[index] = sample;
}


thrust::device_vector<CameraSample> NaiveCameraSampler::genAllSamplesGPU(const CameraObject& camera, FilmObject& film) {
    int width = film.getWidth();
    int height = film.getHeight();
    int count = width*height;
    thrust::device_vector<CameraSample> result(count);
    CameraSample* resultPointer =  thrust::raw_pointer_cast(result.data());

    int numThreads = min(count,MAX_THREADS_PER_BLOCK);
    int numBlocks = divUp(count,numThreads);
    genNaiveSample <<<numBlocks,numThreads>>> (resultPointer,count,width,height);
    CHECK_CUDA_ERROR("gen naive samples");
    return result;
}
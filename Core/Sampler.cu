#include "Sampler.h"
#include "../Samplers/DecideSampleCount.h"
#include "../Samplers/SamplerObject.h"
#include "../Utils/Utils.h"

__global__
void genCameraSamplePixels(CameraSample* resultPointer, int samplesCount, int width, int height, int samplesPerPixel, SamplerObject sampler, int pixelIndexStart) {
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
    sample.x += sampler.rand1(state);
    sample.y += sampler.rand1(state);

    resultPointer[index] = sample;
}

__global__
void genCameraSampleSPPs(CameraSample* resultPointer, int width, int height, int thisSPP, int totalSPP, int completedSPP, SamplerObject sampler) {
    int samplesCount = width * height * thisSPP;
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= samplesCount) {
        return;
    }

    int pixelIndex = index / thisSPP;
    int x = pixelIndex % width;
    int y = pixelIndex / width;

    SamplingState state;
    state.dimension = 0;
    state.index = pixelIndex * totalSPP + index % thisSPP + completedSPP;

    CameraSample sample{ x , y, state };
    sample.x += sampler.rand1(state);
    sample.y += sampler.rand1(state);

    resultPointer[index] = sample;
}


GpuArray<CameraSample> Sampler::genAllCameraSamples(const CameraObject& camera, Film& film, int bytesNeededPerSample, int maxSamplesPerRound) {
    int width = film.width;
    int height = film.height;
    int resolution = film.width * film.height;

    if (cameraSamplingOrder == CameraSamplingOrder::PixelByPixel) {
        unsigned long long sampleCount = decideSampleCount(resolution, samplesPerPixel, completedPixels, bytesNeededPerSample);

        unsigned long long pixelsCount = sampleCount / samplesPerPixel;

        prepare(sampleCount);

        GpuArray<CameraSample> result(sampleCount);

        int numBlocks, numThreads;
        setNumBlocksThreads(sampleCount, numBlocks, numThreads);

        genCameraSamplePixels << <numBlocks, numThreads >> > (result.data, sampleCount, width, height, samplesPerPixel,getObjectFromThis().getCopyForKernel(), completedPixels);
        CHECK_IF_CUDA_ERROR("gen camera samples pixels");

        completedPixels += pixelsCount;
        return result;
    }
    else if (cameraSamplingOrder == CameraSamplingOrder::sppBySpp) {

        int thisSPP = decideSppCount(resolution, samplesPerPixel, completedSPPs, bytesNeededPerSample);

        int count = width * height * thisSPP;

        prepare(count);

        GpuArray<CameraSample> result(count);

        int numBlocks, numThreads;
        setNumBlocksThreads(count, numBlocks, numThreads);

        genCameraSampleSPPs << <numBlocks, numThreads >> > (result.data, width, height, thisSPP, samplesPerPixel,completedSPPs, getObjectFromThis().getCopyForKernel());
        CHECK_IF_CUDA_ERROR("gen camera samples SPPs");

        completedSPPs += thisSPP;

        return result; 

    }
    SIGNAL_ERROR("unrecognized cameraSamplingOrder");
}
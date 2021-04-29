#include "DecideSampleCount.h"
#include "../Utils/GpuCommons.h"



// we might not have enough memory to render all samples at once.
unsigned long long decideSppCount(int resolution, int totalSPP, int completedSPPs, int bytesNeededPerSample) {
    bytesNeededPerSample *= 1.1;

    int sppLeftToDo = totalSPP - completedSPPs;

    size_t freeMemory, totalMemory;
    HANDLE_ERROR(cudaMemGetInfo(&freeMemory, &totalMemory));
    size_t freeMemoryPerPixel = freeMemory / resolution;
    printf("free memory per pixel %d \n", freeMemoryPerPixel);

    int thisSPP = min(sppLeftToDo, (int)(freeMemoryPerPixel / bytesNeededPerSample));


    printf("SPP left:%d,  SPP this round:%d,   SPP completed: %d \n", sppLeftToDo, thisSPP, completedSPPs);

    return thisSPP;
}

unsigned long long decideSampleCount(int resolution, int totalSPP, int completedPixels, int bytesNeededPerSample) {
    bytesNeededPerSample *= 1.1;

    size_t freeMemory, totalMemory;
    HANDLE_ERROR(cudaMemGetInfo(&freeMemory, &totalMemory));

    unsigned long long sampleCount = freeMemory / bytesNeededPerSample;
    unsigned long long pixelCount = sampleCount / totalSPP;
    pixelCount = min((int)pixelCount, resolution - completedPixels);

    sampleCount = pixelCount * totalSPP;

    printf("total pixels:%d,   pixels completed:%d,  pixels this round:%d,   ratio: %f \n",resolution, completedPixels,pixelCount,(float)completedPixels / (float)resolution);
    
    return sampleCount;
}
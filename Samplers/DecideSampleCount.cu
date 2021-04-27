#include "DecideSampleCount.h"
#include "../Utils/GpuCommons.h"


// we might not have enough memory to render all samples at once.
unsigned long long decideSampleCount(Film& film, int totalSPP, int bytesNeededPerSample) {
    unsigned long long resolution = film.width * film.height;

    size_t freeMemory, totalMemory;
    HANDLE_ERROR(cudaMemGetInfo(&freeMemory, &totalMemory));

    unsigned long long sampleCount = freeMemory / bytesNeededPerSample;
    unsigned long long pixelCount = sampleCount / totalSPP;
    pixelCount *= 0.9;
    pixelCount = min(pixelCount, resolution - film.completedPixels);

    sampleCount = pixelCount * totalSPP;

    printf("pixels completed:%d,  pixels this round:%d,   ratio: %f \n", film.completedPixels,pixelCount,(float)film.completedPixels / (float)resolution);
    
    return sampleCount;
}
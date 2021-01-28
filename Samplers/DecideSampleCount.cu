#include "DecideSampleCount.h"
#include "../Utils/GpuCommons.h"


// we might not have enough memory to render all samples at once.



int decideSamplesPerPixel(FilmObject& film, int totalSPP){
    int sppLeftToDo = totalSPP - film.getCompletedSamplesPerPixel();
    int resolution = film.getWidth() * film.getHeight();

    
    size_t freeMemory, totalMemory;
    HANDLE_ERROR(cudaMemGetInfo(&freeMemory, &totalMemory));
    size_t freeMemoryPerPixel = freeMemory / resolution;
    printf("free memory per pixel %d \n", freeMemoryPerPixel);
    
    int thisSPP = min(sppLeftToDo,(int)(freeMemoryPerPixel / 2048));
    
    printf("SPP left:%d,  SPP this round:%d,   SPP completed: %d \n", sppLeftToDo, thisSPP, film.getCompletedSamplesPerPixel());

    film.setCompletedSamplesPerPixel(film.getCompletedSamplesPerPixel() + thisSPP);
    return thisSPP;
}
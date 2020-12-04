#pragma once

#include "../Core/Film.h"
#include "../Films/FilmObject.h"


// we might not have enough memory to render all samples at once.
// so we split the image into regions, and render each region sequentially

#define MAX_SAMPLES_PER_RUN 1024*1024*4


inline int decideSamplesPerPixel(FilmObject& film, int totalSPP){
    int sppLeftToDo = totalSPP - film.getCompletedSamplesPerPixel();
    int resolution = film.getWidth() * film.getHeight();
    int thisSPP = min(sppLeftToDo,divUp(MAX_SAMPLES_PER_RUN,resolution));
    
    printf("%d %d %d \n", sppLeftToDo, thisSPP, film.getCompletedSamplesPerPixel());

    film.setCompletedSamplesPerPixel(film.getCompletedSamplesPerPixel() + thisSPP);
    return thisSPP;
}
#pragma once

#include "RenderResult.h"
#include "Color.h"
#include "CameraSample.h"

class Film{
public:
    virtual RenderResult readCurrentResult()=0;

    __device__
    virtual void addSample(const CameraSample& sample, const Spectrum& color) = 0;

    int width;
    int height;

    int completedSamplesPerPixel = 0;

    int getCompletedSamplesPerPixel(){
        return completedSamplesPerPixel;
    }

    void setCompletedSamplesPerPixel(int spp){
        completedSamplesPerPixel = spp;
    }

    __host__ __device__
    int getWidth() const{
        return width;
    };

    __host__ __device__
    int getHeight() const{
        return height;
    };

};
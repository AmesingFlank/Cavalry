#pragma once

#include "RenderResult.h"
#include "Color.h"
#include "CameraSample.h"

class Film{
public:
    virtual RenderResult readCurrentResult()=0;
    virtual void addSample(const CameraSample& sample, const Spectrum& color) = 0;

    int width;
    int height;

};
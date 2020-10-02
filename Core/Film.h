#pragma once

#include "RenderResult.h"
#include "Color.h"

class Film{
public:
    RenderResult readCurrentResult();
    void addSample(float2 position, Color color);

    int width;
    int height;
};
#pragma once

#include "../Core/Filter.h"

class BoxFilter: public Filter{
public:

    __host__ __device__
    BoxFilter(){
        xwidth = 1;
        ywidth = 1;
    }

    BoxFilter(float xwidth_, float ywidth_) {
        xwidth = xwidth_;
        ywidth = ywidth_;
    }

    __device__
    virtual float contribution(int x, int y, const CameraSample& cameraSample) const {
        if (abs(x - cameraSample.x) > xwidth || abs(y - cameraSample.y) > ywidth) {
            return 0.f;
        }
        return 1.f;
    }
};
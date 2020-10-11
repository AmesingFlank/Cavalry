#pragma once

#include "../Utils/GpuCommons.h"

class Shape;


class VisibilityTest{
public:
    Ray ray;
    float distanceLimit;
    bool useDistanceLimit = false;
    ShapeID sourceGeometry = nullptr;
    ShapeID targetGeometry = nullptr;
    
    __host__ __device__
    void setDistanceLimit(float limit){
        distanceLimit  = limit;
        useDistanceLimit = true;
    }
};
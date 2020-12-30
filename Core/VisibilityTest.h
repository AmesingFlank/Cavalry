#pragma once

#include "../Utils/GpuCommons.h"


class TriangleMesh;

class VisibilityTest{
public:
    Ray ray;
    float distanceLimit;
    bool useDistanceLimit = false;
    int sourceTriangleIndex = -1;
    int targetTriangleIndex = -1;
    
    __host__ __device__
    void setDistanceLimit(float limit){
        distanceLimit  = limit;
        useDistanceLimit = true;
    }
};
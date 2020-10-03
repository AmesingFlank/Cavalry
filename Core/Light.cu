#include "Light.h"
#include "../Utils/MathsCommons.h"
#include "../Utils/GpuCommons.h"


Spectrum EnvironmentMap::sampleRayToPoint(const float3& position,const float2& randomSource, float& outputProbability, Ray& outputRay){
    float3 sampleOnSphere = sampleSphere(randomSource);
    outputRay.origin = position;
    outputRay.direction = sampleOnSphere;
    outputProbability = 1.0 / (4.0*M_PI);

    return evaluateRay(outputRay); 
}

Spectrum EnvironmentMap::evaluateRay(const Ray& ray){
    // to be changed
    return make_float3(0.5*ray.y + 0.5);
}
#include "Light.h"
#include "../Utils/MathsCommons.h"
#include "../Utils/GpuCommons.h"
#include "Primitive.h"

Spectrum EnvironmentMap::sampleRayToPoint(const float3& position,const float2& randomSource, float& outputProbability, Ray& outputRay,VisibilityTest& outputVisibilityTest) const{
    float3 sampleOnSphere = sampleSphere(randomSource);
    outputRay.origin = position;
    outputRay.direction = sampleOnSphere;
    outputProbability = 1.0 / (4.0*M_PI);

    outputVisibilityTest.ray = outputRay;
    return evaluateRay(outputRay); 
}

Spectrum EnvironmentMap::evaluateRay(const Ray& ray) const{
    // to be changed
    return make_float3(0.5*ray.direction.y + 0.5);
}
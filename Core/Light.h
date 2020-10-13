#pragma once

#pragma once

#include "Color.h"
#include "Ray.h"
#include "VisibilityTest.h"
#include "../Utils/MathsCommons.h"

class Light{
public:
    __host__ __device__
    virtual Spectrum sampleRayToPoint(const float3& position,const float2& randomSource, float& outputProbability, Ray& outputRay,VisibilityTest& outputVisibilityTest) const = 0;
};

class AreaLight: public Light{
public:
    __host__ __device__
    virtual Spectrum evaluateRay(const Ray& ray) const = 0;
};

class EnvironmentMap : public AreaLight{
public:
    __host__ __device__
    virtual Spectrum sampleRayToPoint(const float3& position,const float2& randomSource, float& outputProbability, Ray& outputRay,VisibilityTest& outputVisibilityTest) const override{
        float3 sampleOnSphere = sampleSphere(randomSource);
        outputRay.origin = position;
        outputRay.direction = sampleOnSphere;
        outputProbability = 1.0 / (4.0*M_PI);

        outputVisibilityTest.ray = outputRay;
        return EnvironmentMap::evaluateRay(outputRay);
    }

    __host__ __device__
    virtual Spectrum evaluateRay(const Ray& ray) const {
        // to be changed
        return make_float3(0.5*ray.direction.y + 0.5);
    }

};
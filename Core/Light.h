#pragma once

#pragma once

#include "Color.h"

class Light{
public:
    virtual Spectrum sampleRayToPoint(const float3& position,const float2& randomSource, float& outputProbability, Ray& outputRay) const = 0;
};

class AreaLight: public Light{
public:
    virtual Spectrum evaluateRay(const Ray& ray) = 0;
};

class EnvironmentMap : public Light{
public:
    virtual Spectrum sampleRayToPoint(const float3& position,const float2& randomSource, float& outputProbability, Ray& outputRay) override;

    virtual Spectrum evaluateRay(const Ray& ray) override;

};
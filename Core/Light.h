#pragma once

#pragma once

#include "Color.h"
#include "Ray.h"
#include "VisibilityTest.h"

class Light{
public:
    virtual Spectrum sampleRayToPoint(const float3& position,const float2& randomSource, float& outputProbability, Ray& outputRay,VisibilityTest& outputVisibilityTest) const = 0;
};

class AreaLight: public Light{
public:
    virtual Spectrum evaluateRay(const Ray& ray) const = 0;
};

class EnvironmentMap : public AreaLight{
public:
    virtual Spectrum sampleRayToPoint(const float3& position,const float2& randomSource, float& outputProbability, Ray& outputRay,VisibilityTest& outputVisibilityTest) const override;

    virtual Spectrum evaluateRay(const Ray& ray) const ;

};
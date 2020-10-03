#pragma once

#include "../Core/Light.h"
#include "../Utils/GpuCommons.h"

class PointLight: public Light{
public:
    PointLight(float3 position_,Spectrum color_);
    float3 position;
    Spectrum color;

    virtual Spectrum sampleRayToPoint(const float3& position,const float2& randomSource, float& outputProbability, Ray& outputRay, VisibilityTest& outputVisibilityTest) const override;

};
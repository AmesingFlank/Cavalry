#pragma once

#include "../Core/Light.h"
#include "../Utils/GpuCommons.h"

class PointLight: public Light{
public:

    __host__ __device__
    PointLight(){

    }

    __host__ __device__
    PointLight(float3 position_,Spectrum color_):position(position_),color(color_){

    }

    float3 position;
    Spectrum color;

    __device__
    virtual Spectrum sampleRayToPoint(const float3& position, SamplerObject& sampler, float& outputProbability, Ray& outputRay, VisibilityTest& outputVisibilityTest) const override{
        outputProbability = 1;

        outputRay.origin = position;
        outputRay.direction = normalize(this->position - position);

        outputVisibilityTest.ray = outputRay;
        outputVisibilityTest.setDistanceLimit(length(this->position - position));

        return color;
    }

    virtual void buildCpuReferences(const SceneHandle& scene) override {};

    __device__
    virtual void buildGpuReferences(const SceneHandle& scene) override {};

};
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
    virtual Spectrum sampleRayToPoint(const float3& seenFrom, SamplerObject& sampler, float& outputProbability, Ray& outputRay, VisibilityTest& outputVisibilityTest) const override{
        outputProbability = 1;

        outputRay.origin = position;
        outputRay.direction = normalize(this->position - seenFrom);

        outputVisibilityTest.ray = outputRay;
        outputVisibilityTest.setDistanceLimit(length(this->position - seenFrom));

        return color;
    }

    __device__
    virtual float sampleRayToPointPdf(const Ray& sampledRay, const IntersectionResult& lightSurface) const {
        return 1;
    };

    virtual void buildCpuReferences(const SceneHandle& scene) override {};

    __device__
    virtual void buildGpuReferences(const SceneHandle& scene) override {};

};
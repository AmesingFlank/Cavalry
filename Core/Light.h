#pragma once

#pragma once

#include "Color.h"
#include "Ray.h"
#include "VisibilityTest.h"
#include "../Utils/MathsCommons.h"

struct SceneHandle;
class Light{
public:
    
    __host__ __device__
    virtual Spectrum sampleRayToPoint(const float3& position,const float4& randomSource, float& outputProbability, Ray& outputRay,VisibilityTest& outputVisibilityTest) const = 0;

    virtual void buildCpuReferences(const SceneHandle& scene) = 0;

    __device__
    virtual void buildGpuReferences(const SceneHandle& scene) = 0;
    
    virtual void prepareForRender() {};

};

class AreaLight: public Light{
public:
    __host__ __device__
    virtual Spectrum evaluateRay(const Ray& ray) const = 0;
};

class EnvironmentMap : public AreaLight{
public:
    __host__ __device__
    virtual Spectrum sampleRayToPoint(const float3& position,const float4& randomSource, float& outputProbability, Ray& outputRay,VisibilityTest& outputVisibilityTest) const override {
        float3 sampleOnSphere = sampleSphere(to_float2(randomSource));
        outputRay.origin = position;
        outputRay.direction = sampleOnSphere;
        outputProbability = 1.0 / (4.0*M_PI);

        outputVisibilityTest.ray = outputRay;
        return EnvironmentMap::evaluateRay(outputRay);
    }

    __host__ __device__
    virtual Spectrum evaluateRay(const Ray& ray) const override{
        // to be changed
        return make_float3(0.5*ray.direction.y + 0.5) / (4.0 * M_PI);
    }

    virtual void buildCpuReferences(const SceneHandle& scene)override  {};

    __device__
    virtual void buildGpuReferences(const SceneHandle& scene) override {};

    virtual void prepareForRender() {};

};
#pragma once

#pragma once

#include "Color.h"
#include "Ray.h"
#include "VisibilityTest.h"
#include "../Utils/MathsCommons.h"
#include "../Samplers/SamplerObject.h"

struct SceneHandle;
class Light{
public:
    
    __device__
    virtual Spectrum sampleRayToPoint(const float3& seenFrom,SamplerObject& sampler, float& outputProbability, Ray& outputRay,VisibilityTest& outputVisibilityTest) const = 0;

    __device__
    virtual float pdf(const Ray& sampledRay, const IntersectionResult& lightSurface) const = 0;


    virtual void buildCpuReferences(const SceneHandle& scene) = 0;

    __device__
    virtual void buildGpuReferences(const SceneHandle& scene) = 0;
    
    virtual void prepareForRender() {};

};

class AreaLight: public Light{
public:
    __device__
    virtual Spectrum evaluateRay(const Ray& ray) const = 0;
};

class EnvironmentMap : public AreaLight{
public:
    __device__
    virtual Spectrum sampleRayToPoint(const float3& seenFrom, SamplerObject& sampler, float& outputProbability, Ray& outputRay,VisibilityTest& outputVisibilityTest) const override {
        float3 sampleOnSphere = uniformSampleSphere(sampler.rand2());
        outputRay.origin = seenFrom;
        outputRay.direction = sampleOnSphere;
        outputProbability = uniformSampleSpherePdf(sampleOnSphere);

        outputVisibilityTest.ray = outputRay;
        return EnvironmentMap::evaluateRay(outputRay);
    }

    __device__
    virtual float pdf(const Ray& sampledRay, const IntersectionResult& lightSurface) const  {
        return uniformSampleSpherePdf(sampledRay.direction);
    };


    __device__
    virtual Spectrum evaluateRay(const Ray& ray) const override{
        // to be changed
        return make_float3(0.5*ray.direction.y + 0.5) / (4.0 * M_PI);
    }

    virtual void buildCpuReferences(const SceneHandle& scene)override  {};

    __device__
    virtual void buildGpuReferences(const SceneHandle& scene) override {};

    virtual void prepareForRender() {};

};
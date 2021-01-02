#pragma once
#include "../Core/Light.h"


class DiffuseAreaLight:public Light{
public:

    DiffuseAreaLight(Spectrum color_):color(color_){

    }

    DiffuseAreaLight() {};

    Spectrum color;

    int shapeIndex;
    const TriangleMesh* shape;

    virtual void buildCpuReferences(const SceneHandle& scene) override;

    __device__
    virtual void buildGpuReferences(const SceneHandle& scene) override;


    __device__
    virtual Spectrum sampleRayToPoint(const float3& seenFrom, SamplerObject& sampler, float& outputProbability, Ray& outputRay,VisibilityTest& outputVisibilityTest) const override{

        float shapeSampleProbability = 0;

        IntersectionResult shapeSample = shape->sample(seenFrom,sampler,&shapeSampleProbability);
        
        outputProbability = shapeSampleProbability;

        outputRay.origin = seenFrom;
        outputRay.direction = normalize(shapeSample.position - seenFrom);

        outputVisibilityTest.ray = outputRay;
        outputVisibilityTest.targetTriangleIndex = shapeSample.triangleIndex;
        outputVisibilityTest.setDistanceLimit(length(shapeSample.position - seenFrom));

        if (dot(outputRay.direction, shapeSample.normal) >= 0) {
            return make_float3(0, 0, 0);
        }
        return color;
    }

    __device__
    virtual float pdf(const Ray& sampledRay, const IntersectionResult& lightSurface) const {
        return shape->pdf(sampledRay,lightSurface);
    };

    __device__
    virtual Spectrum evaluateRay(const Ray& rayToLight, const IntersectionResult& lightSurface) const{
        if (dot(rayToLight.direction, lightSurface.normal) >= 0) {
            return make_float3(0, 0, 0);
        }
        return color;
    }

};
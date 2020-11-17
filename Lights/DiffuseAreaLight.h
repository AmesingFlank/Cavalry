#pragma once
#include "../Core/Light.h"


class DiffuseAreaLight:public AreaLight{
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
    virtual Spectrum sampleRayToPoint(const float3& position, SamplerObject& sampler, float& outputProbability, Ray& outputRay,VisibilityTest& outputVisibilityTest) const override{


        float shapeSampleProbability = 0;

        

        IntersectionResult shapeSample = shape->sample(position,sampler,&shapeSampleProbability);

        
        outputProbability = shapeSampleProbability;

        outputRay.origin = position;
        outputRay.direction = normalize(shapeSample.position - position);

        outputVisibilityTest.ray = outputRay;
        outputVisibilityTest.targetGeometry = shape->getID();
        outputVisibilityTest.setDistanceLimit(length(shapeSample.position - position));

        if (dot(outputRay.direction, shapeSample.normal) >= 0) {
            return make_float3(0, 0, 0);
        }

        return color;

    }

     __device__
    virtual Spectrum evaluateRay(const Ray& ray) const override{
        return color;
    }

};
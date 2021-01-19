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
    virtual Spectrum sampleRayToPoint(const float3& seenFrom, SamplerObject& sampler, float& outputProbability, Ray& outputRay,VisibilityTest& outputVisibilityTest,IntersectionResult& outputLightSurface) const override{

        float shapeSampleProbability = 0;

        outputLightSurface = shape->sample(seenFrom,sampler,&shapeSampleProbability);
        
        outputProbability = shapeSampleProbability;

        outputRay.origin = seenFrom;
        outputRay.direction = normalize(outputLightSurface.position - seenFrom);

        outputVisibilityTest.ray = outputRay;
        outputVisibilityTest.targetMeshIndex = shape->meshIndex;
        outputVisibilityTest.setDistanceLimit(length(outputLightSurface.position - seenFrom));

        if (dot(outputRay.direction, outputLightSurface.normal) >= 0) {
            return make_float3(0, 0, 0);
        }
        return color;
    }

    // the pdf is for sampleRayToPoint, not sampleRay
    __device__
    virtual float sampleRayToPointPdf(const Ray& sampledRay, const IntersectionResult& lightSurface) const {
        return shape->pdf(sampledRay,lightSurface);
    };

    __device__
    virtual Spectrum evaluateRay(const Ray& rayToLight, const IntersectionResult& lightSurface) const{
        if (dot(rayToLight.direction, lightSurface.normal) >= 0) {
            return make_float3(0, 0, 0);
        }
        return color;
    }

    __device__
    virtual Spectrum sampleRay(SamplerObject& sampler, Ray& outputRay, float3& outputLightNormal, float& outputPositionProbability, float& outputDirectionProbability) const override{

        IntersectionResult shapeSample = shape->sample(make_float3(0,0,0),sampler,&outputPositionProbability,false);

        float3 dir = cosineSampleHemisphere(sampler.rand2());
        outputDirectionProbability = cosineSampleHemispherePdf(dir);

        outputLightNormal = shapeSample.normal;
        outputRay.origin = shapeSample.position;
        outputRay.direction = shapeSample.localToWorld(dir);
        
        return color;
    }

    __device__
    virtual void sampleRayPdf(const Ray& sampledRay, const float3& sampledLightNormal, float& outputPositionProbability, float& outputDirectionProbability) const override{
        float cosine = dot(sampledRay.direction,sampledLightNormal);
        outputDirectionProbability = cosineSampleHemispherePdf(cosine);
        outputDirectionProbability = 1.f / shape->area();
    }

};
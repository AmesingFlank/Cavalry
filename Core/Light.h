#pragma once

#pragma once

#include "Color.h"
#include "Ray.h"
#include "VisibilityTest.h"
#include "../Utils/MathsCommons.h"
#include "../Samplers/SamplerObject.h"
#include "Texture.h"

struct SceneHandle;
class Light{
public:
    
    __device__
    virtual Spectrum sampleRayToPoint(const float3& seenFrom,SamplerObject& sampler, float& outputProbability, Ray& outputRay,VisibilityTest& outputVisibilityTest,IntersectionResult& outputLightSurface) const = 0;

    __device__
    virtual float sampleRayToPointPdf(const Ray& sampledRay, const IntersectionResult& lightSurface) const = 0;

    __device__
    virtual Spectrum sampleRay(SamplerObject& sampler, Ray& outputRay, float3& outputLightNormal, float& outputPositionProbability, float& outputDirectionProbability) const {
        SIGNAL_ERROR("sampleRay not implemented\n");
    }

    __device__
    virtual void sampleRayPdf(const Ray& sampledRay, const float3& sampledLightNormal, float& outputPositionProbability, float& outputDirectionProbability) const {
        SIGNAL_ERROR("sampleRayPdf not implemented\n");
    }


    virtual void buildCpuReferences(const SceneHandle& scene) = 0;

    __device__
    virtual void buildGpuReferences(const SceneHandle& scene) = 0;
    
    virtual void prepareForRender() {};

};


class EnvironmentMap : public Light{
public:
    glm::mat4 lightToWorld;
    glm::mat4 worldToLight;

    Spectrum color;
    bool hasTexture;
    Texture2D texture;

    __host__ __device__
    EnvironmentMap():texture(0,0,true),color(make_float3(0,0,0)),hasTexture(false){}

    __host__ 
    EnvironmentMap(const glm::mat4& lightToWorld_, const Spectrum& color_):
        lightToWorld(lightToWorld_),
        worldToLight(glm::inverse(lightToWorld_)),
        color(color_),
        texture(0,0,true) , hasTexture(false)
    {
        
    }

    __host__
    EnvironmentMap(const glm::mat4& lightToWorld_, const Spectrum& color_, const Texture2D& texture_) :
        lightToWorld(lightToWorld_),
        worldToLight(glm::inverse(lightToWorld_)),
        color(color_),
        texture(texture_), hasTexture(true)
    {

    }


    __device__
    virtual Spectrum sampleRayToPoint(const float3& seenFrom, SamplerObject& sampler, float& outputProbability, Ray& outputRay,VisibilityTest& outputVisibilityTest,IntersectionResult& outputLightSurface) const override {
        float3 sampleOnSphere = uniformSampleSphere(sampler.rand2());
        outputRay.origin = seenFrom;
        outputRay.direction = sampleOnSphere;
        outputProbability = uniformSampleSpherePdf(sampleOnSphere);

        outputVisibilityTest.ray = outputRay;
        return EnvironmentMap::evaluateRay(outputRay);
    }

    __device__
    virtual float sampleRayToPointPdf(const Ray& sampledRay, const IntersectionResult& lightSurface) const  {
        return uniformSampleSpherePdf(sampledRay.direction);
    };


    __device__
    virtual Spectrum evaluateRay(const Ray& ray) const{
        Spectrum result = color;
        if (hasTexture) {
            glm::mat3 directionTransform = glm::mat3(worldToLight);
            float3 dir = directionTransform * ray.direction;

            float phi = atan2(dir.y, dir.x);
            phi = (phi < 0) ? (phi + 2 * M_PI) : phi;

            float theta = acos(clampF(dir.z, -1, 1));

            float2 texCoords = make_float2(phi / (2.f * M_PI), theta / (M_PI));

            float4 texel = texture.readTexture(texCoords);
            Spectrum sampledColor = to_float3(texel);
            result *= sampledColor;
        }

        return result;
    }

    virtual void buildCpuReferences(const SceneHandle& scene)override  {};

    __device__
    virtual void buildGpuReferences(const SceneHandle& scene) override {};

    virtual void prepareForRender() {};

};
#pragma once


#include "../Core/BSDF.h"
#include "../Utils/MathsCommons.h"
#include "../Utils/RandomUtils.h"
#include "Microfacet.h"
#include "Lambertian.h"

class PlasticBSDF: public BSDF{
public:
    
     
    MicrofacetBSDF specular;
    LambertianBSDF diffuse;
     
    __host__ __device__
    PlasticBSDF(){}

    __device__
    PlasticBSDF(const MicrofacetBSDF& specular_, const LambertianBSDF& diffuse_):
    specular(specular_),diffuse(diffuse_)
    {

    }

    __device__
    virtual Spectrum eval(const float3& incident, const float3& exitant) const override {
        return diffuse.eval(incident,exitant) + specular.eval(incident,exitant);
    }

    __device__
    float getSampleDiffuseProbability() const{
        return 0.5f;
    }

    __device__
    virtual Spectrum sample(float2 randomSource, float3& incidentOutput, const float3& exitant, float* probabilityOutput) const {
        float& p = randomSource.x;
        bool sampleDiffuse = p < getSampleDiffuseProbability();
        if(sampleDiffuse){
            p = p * (1.f / getSampleDiffuseProbability());
            diffuse.sample(randomSource,incidentOutput,exitant,probabilityOutput);
        }
        else{
            p = (p - getSampleDiffuseProbability() )* (1.f / (1.f-getSampleDiffuseProbability()));
            specular.sample(randomSource,incidentOutput,exitant,probabilityOutput);
        }
        *probabilityOutput = PlasticBSDF::pdf(incidentOutput,exitant);
        return PlasticBSDF::eval(incidentOutput,exitant);
    }

    __device__
    virtual float pdf(const float3& incident, const float3& exitant) const {
        float diffuseProb = getSampleDiffuseProbability();
        return diffuseProb * diffuse.pdf(incident,exitant) + (1.f-diffuseProb) * specular.pdf(incident,exitant);
    }

    __device__
    virtual bool isDelta() const override { return false; };

};
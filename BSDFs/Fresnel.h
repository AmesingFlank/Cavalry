#pragma once


#include "../Core/BSDF.h"
#include "../Utils/MathsCommons.h"
#include "../Utils/RandomUtils.h"
#include "Fresnel.h"
#include "MicrofacetDistribution.h"

__device__ 
inline float schlick(float cosTheta, float refractiveIndex){
    float f0 = (refractiveIndex-1) / (refractiveIndex + 1);
    f0 = f0*f0;

    float temp = pow5(1- abs(cosTheta));

    return f0 + (1 - f0) * temp;
}

__device__ 
inline Spectrum schlick(float cosTheta, const Spectrum& f0){

    float temp = pow5(1- abs(cosTheta));

    return f0 + (make_float3(1,1,1) - f0) * temp;
}

struct Fresnel{
    float refractiveIndex;
    
    float reflectivityFactor;

    __host__ __device__
    Fresnel(float refractiveIndex_,float reflectivityFactor_ = 1.f):
    refractiveIndex(refractiveIndex_),reflectivityFactor(reflectivityFactor_){}

    __device__
    float eval(float cosTheta) const {
        float f = schlick(cosTheta,refractiveIndex);
        return f * reflectivityFactor;
    }

};



class FresnelBlendBSDF: public BSDF{
public:
    
    Spectrum diffuseColor;
    Spectrum specularColor;
    GGX distribution;

    
	__host__ __device__
	FresnelBlendBSDF(){}

    __host__ __device__
    FresnelBlendBSDF(const Spectrum& diffuse_,const Spectrum& specular_,const GGX& distribution_):
    diffuseColor(diffuse_),specularColor(specular_),distribution(distribution_){
        
    }


    __device__
    virtual Spectrum eval(const float3& incident, const float3& exitant) const override {
        Spectrum diffuse = (28.f / (23.f * M_PI)) * diffuseColor * (make_float3(1,1,1) - specularColor)*
                        (1 - pow5(1 - .5f * abs(cosZenith(incident)) )) *
                        (1 - pow5(1 - .5f * abs(cosZenith(exitant)) ));
        float3 halfVec = incident + exitant;
        if (halfVec.x == 0 && halfVec.y == 0 && halfVec.z == 0) return make_float3(0);
        halfVec = normalize(halfVec);
        Spectrum specular =
            distribution.D(halfVec) * schlick(dot(incident, halfVec), specularColor) /
            (4 * abs(dot(incident, halfVec)) * max(abs(cosZenith(incident)), abs(cosZenith(exitant))));
        return diffuse + specular;
    }

    __device__
    virtual Spectrum sample(float2 randomSource, float3& incidentOutput, const float3& exitant, float* probabilityOutput) const override{
        
        float sampleMicrofacetProb = 0.5;
        bool shouldSampleMicrofacet = randomSource.x < sampleMicrofacetProb;
        if(shouldSampleMicrofacet){
            randomSource.x = randomSource.x * (1.f / sampleMicrofacetProb);
            float3 halfVec = distribution.sample(randomSource,exitant);
            incidentOutput = reflect(exitant, halfVec);
        }
        else{
            randomSource.x =  (randomSource.x-sampleMicrofacetProb) * (1.f / (1.f- sampleMicrofacetProb));
            incidentOutput = cosineSampleHemisphere(randomSource);
            *probabilityOutput = (1.f-sampleMicrofacetProb) * cosineSampleHemispherePdf(incidentOutput);
        }
        *probabilityOutput = 
            (1.f - sampleMicrofacetProb) * cosineSampleHemispherePdf(incidentOutput) + 
            sampleMicrofacetProb * distribution.pdf(incidentOutput);

        return FresnelBlendBSDF::eval(incidentOutput, exitant);

    }

    __device__
    virtual bool isDelta() const override { return false; };

};


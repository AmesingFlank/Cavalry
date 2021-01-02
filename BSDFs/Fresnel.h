#pragma once


#include "../Core/BSDF.h"
#include "../Utils/MathsCommons.h"
#include "../Utils/RandomUtils.h"
#include "Fresnel.h"
#include "MicrofacetDistribution.h"
#include <complex>


__device__ 
inline Spectrum schlick(float cosTheta, const Spectrum& f0){

    float temp = pow5(1- abs(cosTheta));

    return f0 + (make_float3(1,1,1) - f0) * temp;
}

struct Fresnel{
    Spectrum f0;
    
    __device__
    Spectrum eval(float cosTheta) const {
        return schlick(cosTheta,f0);
    }

    static Fresnel createFromIOR(const Spectrum& eta){
        Spectrum temp = (eta-make_float3(1,1,1)) / (eta + make_float3(1,1,1));
        Spectrum f0 = temp * temp;
        return {f0};
    }

    // complex IOR
    static Fresnel createFromIOR(const Spectrum& eta, const Spectrum& k) {

        std::complex<float> etaR(eta.x, k.x);
        std::complex<float> etaG(eta.y, k.y);
        std::complex<float> etaB(eta.z, k.z);

        std::complex<float> one(1, 0);

        float f0R = std::abs(std::pow((etaR - one) / (etaR + one), 2));
        float f0G = std::abs(std::pow((etaG - one) / (etaG + one), 2));
        float f0B = std::abs(std::pow((etaB - one) / (etaB + one), 2));

        return { make_float3(f0R,f0G,f0B) };
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
        if (!sameHemisphere(incident, exitant)) {
            return make_float3(0, 0, 0);
        }
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
    float getSampleMicrofacetProbability() const{
        return 0.5f;
    }

    __device__
    virtual Spectrum sample(float2 randomSource, float3& incidentOutput, const float3& exitant, float* probabilityOutput) const override{
        
        float sampleMicrofacetProb = getSampleMicrofacetProbability();
        bool shouldSampleMicrofacet = randomSource.x < sampleMicrofacetProb;
        float3 halfVec;
        if(shouldSampleMicrofacet){
            randomSource.x = randomSource.x * (1.f / sampleMicrofacetProb);
            halfVec = distribution.sample(randomSource,exitant);
            incidentOutput = reflectF(exitant, halfVec);
            if (!sameHemisphere(incidentOutput, exitant)) {
                return make_float3(0, 0, 0);
            }
        }
        else{
            randomSource.x =  (randomSource.x-sampleMicrofacetProb) * (1.f / (1.f- sampleMicrofacetProb));
            incidentOutput = cosineSampleHemisphere(randomSource);
            halfVec = normalize(incidentOutput + exitant);
            if (exitant.z < 0) {
                incidentOutput.z *= -1;
            }
        }
        
        *probabilityOutput = FresnelBlendBSDF::pdf(incidentOutput,exitant,halfVec);
        
        return FresnelBlendBSDF::eval(incidentOutput, exitant);

    }

    __device__
    float pdf(const float3& incident, const float3& exitant,const float3 halfVec) const {
        float sampleMicrofacetProb = getSampleMicrofacetProbability();
        return (1.f - sampleMicrofacetProb) * cosineSampleHemispherePdf(incident) + 
            sampleMicrofacetProb * distribution.pdf(halfVec,exitant) / (4.f*dot(exitant,halfVec));
    }

    __device__
    virtual float pdf(const float3& incident, const float3& exitant) const {
        float3 halfVec = normalize(incident + exitant);
        return FresnelBlendBSDF::pdf(incident,exitant,halfVec);
    }


    __device__
    virtual bool isDelta() const override { return false; };

};


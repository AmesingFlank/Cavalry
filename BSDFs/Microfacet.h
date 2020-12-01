#pragma once
#include "../Core/BSDF.h"


#include "../Utils/MathsCommons.h"
#include "../Utils/RandomUtils.h"



class MicrofacetBSDF: public BSDF{
public:
    Spectrum color;
    GGX distribution;
    Fresnel fresnel;

	__host__ __device__
	MicrofacetBSDF():fresnel(1){}

    __host__ __device__
    MicrofacetBSDF(const Spectrum& color_,const GGX& distribution_,const Fresnel& fresnel_):
    color(color_),distribution(distribution_),fresnel(fresnel_){

    }

    __device__
    virtual Spectrum eval(const float3& incident, const float3& exitant) const override {
        float cosThetaO = abs(cosZenith(exitant));
        float cosThetaI = abs(cosZenith(incident));

        float3 halfVec = incident + exitant; 

        if (cosThetaI == 0 || cosThetaO == 0) return make_float3(0,0,0); 
        if (halfVec.x == 0 && halfVec.y == 0 && halfVec.z == 0) return make_float3(0,0,0); 
        
        halfVec = normalize(halfVec); 
        float F = fresnel.eval(dot(exitant, halfVec)); 

        return color * distribution.D(halfVec) * distribution.G(exitant, incident)*F/ (4 * cosThetaI * cosThetaO);
    }

    __device__
    virtual Spectrum sample(float2 randomSource, float3& incidentOutput, const float3& exitant, float* probabilityOutput) const {
        

        incidentOutput = cosineSampleHemisphere(randomSource);
        *probabilityOutput = cosineSampleHemispherePdf(incidentOutput);


        return MicrofacetBSDF::eval(incidentOutput, exitant);

    }

};


#pragma once


#include "../Core/BSDF.h"
#include "../Utils/MathsCommons.h"
#include "../Utils/RandomUtils.h"
 

class SpecularBSDF: public BSDF{
public:
    Spectrum reflectionColor;
    
    Spectrum transmissionColor;

    Fresnel fresnel;

    float aboveIOR;
    float belowIOR;

    bool hasTransmission;

	__host__ __device__
	SpecularBSDF(){}

    

    __host__ __device__
    SpecularBSDF(const Spectrum& reflectionColor_,const Fresnel& fresnel_, 
    bool hasTransmission_ = false, const Spectrum& transmissionColor_ = make_float3(0,0,0), float aboveIOR_ = 1.5f, float belowIOR_ = 1.5f):
    reflectionColor(reflectionColor_),fresnel(fresnel_),
    hasTransmission(hasTransmission_), transmissionColor(transmissionColor_),aboveIOR(aboveIOR_),belowIOR(belowIOR_)
    {

    }
    

    // assuming incident and exitant are valid
    __device__
    Spectrum evalReflection(const float3& incident, const float3& exitant) const{
        auto F = fresnel.eval(abs(cosZenith(incident))); 
        return reflectionColor *F / abs(cosZenith(incident));
    }
    

    __device__
    Spectrum sampleReflection(float2 randomSource, float3& incidentOutput, const float3& exitant, float* probabilityOutput) const {
        
        incidentOutput = make_float3(-exitant.x,-exitant.y,exitant.z);
        *probabilityOutput = 1;

        return evalReflection(incidentOutput, exitant);
    }
    

    // assuming incident and exitant are valid
    __device__
    Spectrum evalTransmission(const float3& incident, const float3& exitant) const {
        float3 F = fresnel.eval(abs(cosZenith(incident))); 
        //printf("evaling transmission:  %f %f %f,      %f %f %f\n",  XYZ(transmissionColor),XYZ(F));
        return transmissionColor *(make_float3(1,1,1)-F) / abs(cosZenith(incident));
    }

    __device__
    Spectrum sampleTransmission(float2 randomSource, float3& incidentOutput, const float3& exitant, float* probabilityOutput) const {
        
        float eta = cosZenith(exitant) > 0 ? (aboveIOR / belowIOR): (belowIOR / aboveIOR) ;
        float3 normal = make_float3(0, 0, 1);
        if (exitant.z < 0) normal.z = -1;       
        
        if (!computeRefraction(exitant, normal, eta, incidentOutput)){
            //printf("samping trans %f %f %f,  normal:%f %f %f,      eta: %f.  Got total internal\n", XYZ(exitant),XYZ(normal), eta);
            return make_float3(0,0,0);
        }
        else {
            //printf("samping trans %f %f %f,  normal:%f %f %f,      eta: %f.  Good\n", XYZ(exitant), XYZ(normal), eta);
        }
        
        *probabilityOutput = 1;

        return evalTransmission(incidentOutput, exitant);
    }

   
    __device__
    virtual Spectrum eval(const float3& incident, const float3& exitant) const override {
        if (!hasTransmission || sameHemisphere(incident,exitant)) {
            if(!(incident.x == -exitant.x && incident.y== -exitant.y && incident.z==exitant.z)){
                return make_float3(0,0,0);
            }
            return evalReflection(incident, exitant);
        }
        else{
            float eta = cosZenith(exitant) > 0 ? (aboveIOR / belowIOR) : (belowIOR / aboveIOR);
            float3 refraction;
            float3 normal = make_float3(0, 0, 1);
            if (exitant.z < 0) normal.z = -1;
            if (!computeRefraction(exitant,normal, eta, refraction)){
                return make_float3(0,0,0);
            }
            if(!(refraction==incident)){
                return make_float3(0,0,0);
            }
            return evalTransmission(incident,exitant);
        }
    }
    
    __device__
    virtual Spectrum sample(float2 randomSource, float3& incidentOutput, const float3& exitant, float* probabilityOutput) const {
        if(!hasTransmission){
            return sampleReflection(randomSource,incidentOutput,exitant,probabilityOutput);
        }

        float sampleReflectionProbability = fresnel.eval(abs(cosZenith(exitant))).x;
        if (isAllZero(reflectionColor)) {
            sampleReflectionProbability = 0;
        }
        else if (isAllZero(transmissionColor)) {
            sampleReflectionProbability = 1;
        }

        bool useBRDF = randomSource.x < sampleReflectionProbability;
        if(useBRDF){
            randomSource.x = randomSource.x*(1.f/sampleReflectionProbability);
            Spectrum result = sampleReflection(randomSource,incidentOutput,exitant,probabilityOutput);
            *probabilityOutput *= sampleReflectionProbability;
            return result;
        }
        else{
            randomSource.x = (randomSource.x-sampleReflectionProbability)*(1.f/(1.f-sampleReflectionProbability));
            Spectrum result = sampleTransmission(randomSource,incidentOutput,exitant,probabilityOutput);
            *probabilityOutput *= (1.f - sampleReflectionProbability);
            return result;
        }
    }
    
    __device__
    virtual bool isDelta() const override { return true; };
    
};




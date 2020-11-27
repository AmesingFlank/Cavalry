#pragma once


#include "../Core/BSDF.h"
#include "../Utils/MathsCommons.h"
#include "../Utils/RandomUtils.h"

__device__ 
inline float schlick(float cosTheta, float refractiveIndex){
    float f0 = (refractiveIndex-1) / (refractiveIndex + 1);
    f0 = f0*f0;

    float temp = 1- abs(cosTheta);
    temp = temp*temp*temp*temp*temp;

    return f0 + (1 - f0) * temp;
}

class FresnelBSDF: public BSDF{
public:
    
    float refractiveIndex;
    
    float reflectivityFactor;
    
	__host__ __device__
	FresnelBSDF(){}

    __host__ __device__
    FresnelBSDF(float refractiveIndex_,float reflectivityFactor_ = 1.f):
    refractiveIndex(refractiveIndex_),reflectivityFactor(reflectivityFactor_){}

    __device__
    virtual Spectrum eval(const float3& incident, const float3& exitant) const override {
        if(!(incident.x == -exitant.x && incident.y== -exitant.y && incident.z==exitant.z)){
            return make_float3(0,0,0);
        }
        float f = schlick(exitant.z,refractiveIndex);
        return make_float3(f,f,f) * reflectivityFactor;
    }

    __device__
    virtual Spectrum sample(const float2& randomSource, float3& incidentOutput, const float3& exitant, float* probabilityOutput) const {
        

        incidentOutput = exitant*-1;
        incidentOutput.z = exitant.z;


        *probabilityOutput = 1;

        return FresnelBSDF::eval(incidentOutput, exitant);

    }

    __device__
    virtual bool isDelta() const override { return true; };

};


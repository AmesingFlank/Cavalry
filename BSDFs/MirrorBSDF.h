#pragma once


#include "../Core/BSDF.h"
#include "../Utils/MathsCommons.h"
#include "../Utils/RandomUtils.h"
 

class MirrorBSDF: public BSDF{
public:
    
     
    float reflectivityFactor;
     
    __host__ __device__
    MirrorBSDF(float reflectivityFactor_ = 1.f): reflectivityFactor(reflectivityFactor_){}

    __device__
    virtual Spectrum eval(const float3& incident, const float3& exitant) const override {
        if(!(incident.x == -exitant.x && incident.y== -exitant.y && incident.z==exitant.z)){
            return make_float3(0,0,0);
        }
        return make_float3(1,1,1) * reflectivityFactor;
    }

    __device__
    virtual Spectrum sample(float2 randomSource, float3& incidentOutput, const float3& exitant, float* probabilityOutput) const {
        

        incidentOutput = exitant*-1;
        incidentOutput.z = exitant.z;


        *probabilityOutput = 1;

        return MirrorBSDF::eval(incidentOutput, exitant);

    }

    __device__
    virtual bool isDelta() const override { return true; };

};


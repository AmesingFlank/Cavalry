#pragma once
#include "../Core/BSDF.h"
#include "../Utils/MathsCommons.h"


class LambertianBSDF: public BSDF{
public:
    Spectrum baseColor;


	__host__ __device__
	LambertianBSDF(){}

    __host__ __device__
    LambertianBSDF(const Spectrum& baseColor_):baseColor(baseColor_){}

    __device__
    virtual Spectrum eval(const float3& incident, const float3& exitant) const override {
         return baseColor / M_PI;
    }

    __device__
    virtual Spectrum sample(const float2& randomSource, float3& incidentOutput, const float3& exitant, float* probabilityOutput) const {
        float angle0 = randomSource.x * 2* M_PI;
        float angle1 = randomSource.y * M_PI / 2.f;
        float x = cos(angle0) * cos(angle1);
        float y = sin(angle1);
        float z = sin(angle0) * cos(angle1);
        incidentOutput = make_float3(x, y, z);
        if (dot(incidentOutput, exitant) < 0) {
            incidentOutput *= -1;
        }
        *probabilityOutput = 1.f / (2 * M_PI);
        return eval(incidentOutput, exitant);

    }

};


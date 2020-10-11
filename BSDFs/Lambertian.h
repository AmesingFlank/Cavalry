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

    __host__ __device__
    virtual Spectrum eval(const float3& incident, const float3& exitant) const override {
         return baseColor / M_PI;
    }
};


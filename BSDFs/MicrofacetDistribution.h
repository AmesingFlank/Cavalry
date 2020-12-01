#pragma once

#include "../Core/BSDF.h"
#include "../Utils/MathsCommons.h"
#include "../Utils/RandomUtils.h"


struct GGX {

    float alpha_x, alpha_y;

    // normal distribution (normal is half vector)
    __device__
    float D(const float3& normal) const{
        if(dot(normal,make_float3(0,0,1)) == 0){
            return 0;
        }
        float tanSquaredTheta = tanSquaredZenith(normal);

        const float cos4Theta = cosSquaredZenith(normal) * cosSquaredZenith(normal);

        float e = 
            (cosSquaredAzimuth(normal) / (alpha_x * alpha_x) + 
            sinSquaredAzimuth(normal) / (alpha_y * alpha_y)) 
            * tanSquaredTheta;

        return 1 / (M_PI * alpha_x * alpha_y * cos4Theta * (1 + e) * (1 + e));
    }

    __device__
    float Lambda(const float3& w) const {
        if(dot(w,make_float3(0,0,1)) == 0){
            return 0;
        }
    
        float absTanTheta = abs(tanZenith(w));
        float alpha = sqrt(cosSquaredAzimuth(w) * alpha_x * alpha_x + sinSquaredAzimuth(w) * alpha_y * alpha_y);
        float alpha2Tan2Theta = (alpha * absTanTheta) * (alpha * absTanTheta);

        return (-1 + std::sqrt(1.f + alpha2Tan2Theta)) / 2;
    }

    __device__
    float G(const float3& incident, const float3& exitant) const {
        return 1.f/(1.f+ Lambda(incident) + Lambda(exitant));
    }

    // alpha is sqrt(2) * RMS slope of microfacets
    // roughness is [0,1]
    __device__ __host__
    static float roughnessToAlpha(float roughness){
        roughness = max(roughness, (float)1e-3);
        float x = log(roughness);
        return 1.62142f + 0.819955f * x + 0.1734f * x * x + 0.0171201f * x * x * x +
            0.000640711f * x * x * x * x;
    }

    __device__
    float pdf(const float3& normal)const {
        return D(normal) * abs(cosZenith(normal));
    }

    // sample a microfacet normal.
    __device__
    float3 sample(float2 randomSource, const float3& exitant)const {
        float3 exitantStretched =
        normalize(make_float3(alpha_x * exitant.x, alpha_y * exitant.y, exitant.z));

        float slope_x, slope_y;
        GGX::sampleGGX11(cosZenith(exitantStretched), randomSource, &slope_x, &slope_y);

        //rotate
        float temp = cosAzimuth(exitantStretched) * slope_x - sinAzimuth(exitantStretched) * slope_y;
        slope_y = sinAzimuth(exitantStretched) * slope_x + cosAzimuth(exitantStretched) * slope_y;
        slope_x = temp;

        //unstretch
        slope_x = alpha_x * slope_x;
        slope_y = alpha_y * slope_y;

        float3 result = normalize(make_float3(-slope_x, -slope_y, 1.));
        return result;
    }

    __device__
    static void sampleGGX11(float cosTheta, float2 randomSource,float *slope_x, float *slope_y) {
        float& U1 = randomSource.x;
        float& U2 = randomSource.y;
        // special case (normal incidence)
        if (cosTheta > .9999) {
            float r = sqrt(U1 / (1 - U1));
            float phi = 6.28318530718 * U2;
            *slope_x = r * cos(phi);
            *slope_y = r * sin(phi);
            return;
        }

        float sinTheta =
            sqrt(max((float)0, (float)1 - cosTheta * cosTheta));
        float tanTheta = sinTheta / cosTheta;
        float a = 1 / tanTheta;
        float G1 = 2 / (1 + sqrt(1.f + 1.f / (a * a)));

        // sample slope_x
        float A = 2 * U1 / G1 - 1;
        float tmp = 1.f / (A * A - 1.f);
        if (tmp > 1e10) tmp = 1e10;
        float B = tanTheta;
        float D = sqrt(max(float(B * B * tmp * tmp - (A * A - B * B) * tmp), float(0)));
        float slope_x_1 = B * tmp - D;
        float slope_x_2 = B * tmp + D;
        *slope_x = (A < 0 || slope_x_2 > 1.f / tanTheta) ? slope_x_1 : slope_x_2;

        // sample slope_y
        float S;
        if (U2 > 0.5f) {
            S = 1.f;
            U2 = 2.f * (U2 - .5f);
        } else {
            S = -1.f;
            U2 = 2.f * (.5f - U2);
        }
        float z =
            (U2 * (U2 * (U2 * 0.27385f - 0.73369f) + 0.46341f)) /
            (U2 * (U2 * (U2 * 0.093073f + 0.309420f) - 1.000000f) + 0.597999f);
        *slope_y = S * z * sqrt(1.f + *slope_x * *slope_x);

    }

};

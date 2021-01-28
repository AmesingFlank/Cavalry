#pragma once

#include "../Utils/GpuCommons.h"
#include "../Utils/MathsCommons.h"
#include "TransportMode.h"

using Spectrum = float3;


inline Spectrum colorTemperatureToRGB(float kelvin){

    float temp = kelvin / 100.f;

    float red, green, blue;

    if( temp <= 66 ){ 

        red = 255; 
        
        green = temp;
        green = 99.4708025861 * log(green) - 161.1195681661;
        
        if( temp <= 19){

            blue = 0;

        } else {

            blue = temp-10;
            blue = 138.5177312231 * log(blue) - 305.0447927307;
        }

    } else {

        red = temp - 60;
        red = 329.698727446 * pow(red, -0.1332047592);
        
        green = temp - 60;
        green = 288.1221695283 * pow(green, -0.0755148492 );

        blue = 255;

    }

    red = min(255.f,max(0.f,red));
    green = min(255.f,max(0.f,green));
    blue = min(255.f,max(0.f,blue));

    return make_float3(red,green,blue) / 255.f;
}


__host__ __device__
inline float gammaCorrect(float value) {
    if (value <= 0.0031308f) 
        return 12.92f * value;
    return 1.055f * pow(value, (float)(1.f / 2.4f)) - 0.055f;
}

__host__ __device__
inline float inverseGammaCorrect(float value) {
    if (value <= 0.04045f) 
        return value * 1.f / 12.92f;
    return pow((value + 0.055f) * 1.f / 1.055f, (float)2.4f);
}


__host__ __device__
inline Spectrum clampBetween0And1(const Spectrum& spectrum) {
    float r = max(0.f, min(1.f, spectrum.x));
    float g = max(0.f, min(1.f, spectrum.y));
    float b = max(0.f, min(1.f, spectrum.z));
    return make_float3(r, g, b);
}

__host__ __device__
inline void writeColorAt(const Spectrum&  color, unsigned char* address){
    address[0] = gammaCorrect(color.x) * 255;
    address[1] = gammaCorrect(color.y) * 255;
    address[2] = gammaCorrect(color.z) * 255;
}

__device__
inline void atomicAdd(Spectrum* target, Spectrum toAdd){
    atomicAdd(&(target->x),toAdd.x);
    atomicAdd(&(target->y),toAdd.y);
    atomicAdd(&(target->z),toAdd.z);
}


__device__ __host__
inline bool isAllZero(const Spectrum& color) {
    return color.x == 0 && color.y == 0 && color.z == 0;
}

__device__ __host__
inline float luminance(const Spectrum& color){
    return  dot(color, make_float3(0.212671f, 0.715160f, 0.072169f));
}
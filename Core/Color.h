#pragma once

#include "../Utils/GpuCommons.h"
#include "../Utils/MathsCommons.h"


using Spectrum = float3;


__host__ __device__
inline Spectrum clampBetween0And1(const Spectrum& spectrum) {
    float r = max(0.f, min(1.f, spectrum.x));
    float g = max(0.f, min(1.f, spectrum.y));
    float b = max(0.f, min(1.f, spectrum.z));
    return make_float3(r, g, b);
}

__host__ __device__
inline void writeColorAt(const Spectrum&  color, unsigned char* address){
    address[0] = color.x * 255;
    address[1] = color.y * 255;
    address[2] = color.z * 255;
}
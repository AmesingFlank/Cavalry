#pragma once

#include "../Utils/GpuCommons.h"

using Color = float3;
using Spectrum = float3;

inline void writeColorAt(const Color& color, unsigned char* address){
    address[0] = color.x * 255;
    address[1] = color.y * 255;
    address[2] = color.z * 255;
}
#pragma once

#include "../Utils/GpuCommons.h"
#include "../Utils/MathsCommons.h"


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
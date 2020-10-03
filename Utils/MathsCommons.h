#pragma once

#include <helper_math.h>
#include <device_launch_parameters.h>

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <math.h>

inline float3 sampleSphere(const float2& randomSource){
    float u = randomSource.x * 2 * M_PI;
    float v = (randomSource.y - 0.5) * M_PI;
    return make_float3(
		cos(v)*cos(u),
		sin(v),
		cos(v)*sin(u)
	);
}


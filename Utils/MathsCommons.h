#pragma once

#include <helper_math.h>
#include <device_launch_parameters.h>

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <math.h>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp> 
#include <vector>

template <typename T>
__host__ __device__
inline float3 to_float3(const T& vec){
	return make_float3(vec.x,vec.y,vec.z);
}


template <typename T>
__host__ __device__
inline glm::vec3 to_vec3(const T& vec){
	return glm::vec3(vec.x,vec.y,vec.z);
}

inline glm::mat4 to_mat4(std::vector<float> data) {
	glm::mat4 result;
	memcpy(&result, data.data(), 16 * sizeof(float));
	return result;
}


inline float3 sampleSphere(const float2& randomSource){
    float u = randomSource.x * 2 * M_PI;
    float v = (randomSource.y - 0.5) * M_PI;
    return make_float3(
		cos(v)*cos(u),
		sin(v),
		cos(v)*sin(u)
	);
}


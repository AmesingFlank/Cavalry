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
inline float2 to_float2(const T& vec) {
	return make_float2(vec.x, vec.y);
}


template <typename T>
__host__ __device__
inline glm::vec2 to_vec2(const T& vec) {
	return glm::vec3(vec.x, vec.y);
}


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

template <typename T>
__host__ __device__
inline glm::mat3 buildMat3UsingVecsAsRows(const T& v1, const T& v2, const T& v3) {
	return glm::mat3{ 
		v1.x, v2.x, v3.x, 
		v1.y, v2.y, v3.y,  
		v1.z, v2.z, v3.z };
}

template <typename T>
__host__ __device__
inline glm::mat3 buildMat3UsingVecsAsCols(const T& v1, const T& v2, const T& v3) {
	return glm::mat3{
		v1.x, v1.y, v1.z,
		v2.x, v2.y, v2.z,
		v3.x, v3.y, v3.z, };
}

__host__ __device__
inline float lengthSquared(const float3& v) {
	return v.x * v.x + v.y * v.y + v.z * v.z;
}

__host__ __device__
inline float3 apply(const glm::mat4& transform, const float3& v){
	glm::vec4 transformed = transform * glm::vec4(to_vec3(v),1.f);
	return make_float3(transformed.x / transformed.w, transformed.y / transformed.w, transformed.z / transformed.w );
}

__host__ __device__
inline glm::mat3 getTransformForNormal(const glm::mat4& transform){
	return glm::transpose(glm::inverse(glm::mat3(transform)));
}

__host__ __device__
inline float3 operator* (const glm::mat3& mat, const float3& v){
	return to_float3(mat*to_vec3(v));
}

#define XYZ(v) v.x , v.y , v.z



__device__ __inline__ int   min_min   (int a, int b, int c) { int v; asm("vmin.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   min_max   (int a, int b, int c) { int v; asm("vmin.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   max_min   (int a, int b, int c) { int v; asm("vmax.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   max_max   (int a, int b, int c) { int v; asm("vmax.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ float fmin_fmin (float a, float b, float c) { return __int_as_float(min_min(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __inline__ float fmin_fmax (float a, float b, float c) { return __int_as_float(min_max(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __inline__ float fmax_fmin (float a, float b, float c) { return __int_as_float(max_min(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __inline__ float fmax_fmax (float a, float b, float c) { return __int_as_float(max_max(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }


__device__
inline float clampF(float x, float minimum,float maximum){
	return fmaxf(minimum,fminf(x,maximum));
}

__device__
inline int clampF(int x, int minimum,int maximum){
	return max(minimum,min(x,maximum));
}



//theta is the zenithAngle
//phi is the aziumth angle

__device__ inline float cosZenith(const float3 &w) { return w.z; }
__device__ inline float cosSquaredZenith(const float3 &w) { return w.z * w.z; }

__device__ inline float sinSquaredZenith(const float3 &w) {
    return max((float)0, (float)1 - cosSquaredZenith(w));
}

__device__ inline float sinZenith(const float3 &w) { return sqrt(sinSquaredZenith(w)); }

__device__ inline float tanZenith(const float3 &w) { return sinZenith(w) / cosZenith(w); }

__device__ inline float tanSquaredZenith(const float3 &w) {
    return sinSquaredZenith(w) / cosSquaredZenith(w);
}

__device__ inline float cosAzimuth(const float3 &w) {
    float sinTheta = sinZenith(w);
    return (sinZenith == 0) ? 1 : clampF(w.x / sinTheta, -1.f, 1.f);
}

__device__ inline float sinAzimuth(const float3 &w) {
    float sinTheta = sinZenith(w);
    return (sinZenith == 0) ? 0 : clampF(w.y / sinTheta, -1.f, 1.f);
}

__device__ inline float cosSquaredAzimuth(const float3 &w) { return cosAzimuth(w) * cosAzimuth(w); }

__device__ inline float sinSquaredAzimuth(const float3 &w) { return sinAzimuth(w) * sinAzimuth(w); }


__device__
inline bool sameHemisphere(const float3 &v1, const float3 &v2) {
    return v1.z * v2.z > 0;
}

__device__
inline bool sameSign(float f1, float f2) {
	return f1*f2 > 0;
}

__device__
inline float pow5(float f){
	return (f*f)*(f*f)*f;
}


__host__ __device__
inline float3 reflectF(const float3& incident, const float3& normal)
{
	return 2.0f * normal * dot(normal, incident) - incident;
}

// Theta: Zenith Angle
__device__ // returns whether refraction is possible (i.e. whether there is total internal reflection)
inline bool computeRefraction(const float3 &incident, const float3 &normal, float IOR,
                    float3& exitantOutput) {
    // Compute $\cos \theta_\roman{t}$ using Snell's law
    float cosThetaIncident = dot(normal, incident);
    float sin2ThetaIncident = max(0.f, 1.f - cosThetaIncident * cosThetaIncident);
    float sin2ThetaExitant = IOR * IOR * sin2ThetaIncident;

    // Handle total internal reflection for transmission
    if (sin2ThetaExitant >= 1) return false;
    float cosThetaExitant = sqrt(1 - sin2ThetaExitant);
    exitantOutput = IOR * (incident * -1.f) + (IOR * cosThetaIncident - cosThetaExitant) * normal;
    return true;
}

__device__ 
inline bool operator==(const float3 &v1, const float3& v2) {
    // Compute $\cos \theta_\roman{t}$ using Snell's law
    return v1.x==v2.x && v1.y==v2.y && v1.z==v2.z;
}


__device__
inline void buildCoordinateSystem(const float3& normal, float3& tangent0, float3& tangent1) {
    if (abs(normal.x) < 0.5) {
        tangent0 = cross(make_float3(1, 0, 0), normal);
    }
    else {
        tangent0 = cross(make_float3(0, 1, 0), normal);
    }
    tangent0 = normalize(tangent0);
    tangent1 = normalize(cross(normal, tangent0));
}
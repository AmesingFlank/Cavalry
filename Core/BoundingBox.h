#pragma once

#include "../Utils/MathsCommons.h"
#include "../Utils/GpuCommons.h"
#include "Ray.h"



struct AABB{
    float3 minimum;
    float3 maximum;

    __host__ __device__
    float3 centroid(){
        return (minimum+maximum)*0.5f;
    }

    __host__ __device__
    float3 extent(){
        return maximum - minimum;
    }

    __host__ __device__
    bool contains(const float3& point) {
        return minimum.x <= point.x && point.x <= maximum.x &&
            minimum.y <= point.y && point.y <= maximum.y &&
            minimum.z <= point.z && point.z <= maximum.z;
    }

    __device__
    bool intersect_naive(const Ray& r,float& minDist,const float3& invDir){
        
		float3 tmin = (minimum - r.origin) / r.direction;
		float3 tmax = (maximum - r.origin) / r.direction;

		float3 realMin = fminf(tmin, tmax);
		float3 realMax = fmaxf(tmin, tmax);

		float minmax = fminf(fminf(realMax.x, realMax.y), realMax.z);
		float maxmin = fmaxf(fmaxf(realMin.x, realMin.y), realMin.z);

        if (maxmin <= minmax) {
            //float epsilon = 0.001f; // required to prevent self intersection
            if(minmax > 0){
                minDist = fmaxf(0.f,maxmin);
                return true;
            }
            // otherwise maxmin<0 and minmax<0, so no intersection
        }
        minDist = -1; // indicates no-intersection
		return false;
    }

    // optimization technqiue : Understanding the Efficiency of Ray Traversal on GPUs – Kepler and Fermi Addendum
    __device__
    bool intersect(const Ray& r,float& minDist,const float3& invDir){
        
		float3 tmin = (minimum - r.origin) * invDir;
		float3 tmax = (maximum - r.origin) * invDir;

		float maxmin = fmin_fmax(tmin.x,tmax.x,fmin_fmax(tmin.y,tmax.y, fminf(tmin.z,tmax.z)));
        float minmax = fmax_fmin(tmin.x,tmax.x,fmax_fmin(tmin.y,tmax.y, fmaxf(tmin.z,tmax.z)));

		if (maxmin <= minmax) {
            //float epsilon = 0.001f; // required to prevent self intersection
            if(minmax > 0){
                minDist = fmaxf(0.f,maxmin);
                return true;
            }
            // otherwise maxmin<0 and minmax<0, so no intersection
        }
        minDist = -1; // indicates no-intersection
		return false;
    }

    __device__
    float computeSurfaceArea(){
        float3 extent = maximum - minimum;
        return 2.f*(extent.x * extent.y + extent.y*extent.z + extent.x*extent.z);
    }
};

__host__ __device__
inline AABB unionBoxes(const AABB& a,const AABB& b){
    return {
        make_float3(
            fminf(a.minimum.x,b.minimum.x),
            fminf(a.minimum.y,b.minimum.y),
            fminf(a.minimum.z,b.minimum.z)
        ),
        make_float3(
            fmaxf(a.maximum.x,b.maximum.x),
            fmaxf(a.maximum.y,b.maximum.y),
            fmaxf(a.maximum.z,b.maximum.z)
        )
    };
};
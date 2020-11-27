#pragma once

#include "../Utils/MathsCommons.h"
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

    __host__ __device__
    bool intersect(const Ray& r,float& minDist){
        
		float3 tmin = (minimum - r.origin) / r.direction;
		float3 tmax = (maximum - r.origin) / r.direction;

		float3 realMin = fminf(tmin, tmax);
		float3 realMax = fmaxf(tmin, tmax);

		float minmax = min(min(realMax.x, realMax.y), realMax.z);
		float maxmin = max(max(realMin.x, realMin.y), realMin.z);

        if (contains(r.origin)) {
            minDist = 0;
            return true;
        }

        
		if (minmax >= maxmin) { 
            //float epsilon = 0.001f; // required to prevent self intersection
            if (maxmin > 0) {
                minDist = maxmin;
                return true;
            }
        }
        minDist = -1; // indicates no-intersection
		return false;
    }
};

__host__ __device__
inline AABB unionBoxes(const AABB& a,const AABB& b){
    return {
        make_float3(
            min(a.minimum.x,b.minimum.x),
            min(a.minimum.y,b.minimum.y),
            min(a.minimum.z,b.minimum.z)
        ),
        make_float3(
            max(a.maximum.x,b.maximum.x),
            max(a.maximum.y,b.maximum.y),
            max(a.maximum.z,b.maximum.z)
        )
    };
};
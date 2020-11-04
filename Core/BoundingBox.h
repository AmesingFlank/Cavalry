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
    bool intersect(const Ray& r){
        

		float3 tmin = (minimum - r.origin) / r.direction;
		float3 tmax = (maximum - r.origin) / r.direction;

		float3 real_min = fminf(tmin, tmax);
		float3 real_max = fmaxf(tmin, tmax);

		float minmax = min(min(real_max.x, real_max.y), real_max.z);
		float maxmin = max(max(real_min.x, real_min.y), real_min.z);

        
		if (minmax >= maxmin) { 
            //float epsilon = 0.001f; // required to prevent self intersection
            //return maxmin > epsilon;
            return maxmin > 0;
        }
		else return false;
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
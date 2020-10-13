#pragma once

#include "../Core/Camera.h"

class PerspectiveCamera: public Camera{
public:
	__host__ __device__
	Ray genRay(const CameraSample& cameraSample) const
	{
		float3 pixelLocation = {0,0,0};
		pixelLocation.x = cameraSample.x - 0.5;
		pixelLocation.y = 0.5 - cameraSample.y;

		float3 origin = {0,0,-2};
		Ray ray;
		ray.origin = origin;
		ray.direction = normalize(pixelLocation - origin);
		return ray;

	}
};
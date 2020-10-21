#pragma once

#include "../Core/Camera.h"
#include "../Utils/MathsCommons.h"

class PerspectiveCamera: public Camera{
public:

	float3 eye;
	float3 center;
	float3 up;
	float fovX;
	float fovY;
	int filmWidth;
	int filmHeight;

	// depth value (z value) of a hypothetical film,
	// placed at the position which makes the pixels have size 1x1 in world space.
	float depthForUnitSizePixel;

	glm::mat4 lookAtInv;

	__host__ 
	PerspectiveCamera();

	__host__
	PerspectiveCamera(const float3& eye_, const float3& center_, const float3& up_, float fov_, int filmWidth_, int filmHeight_);



	__host__ __device__
	Ray genRay(const CameraSample& cameraSample) const
	{
		float3 pixelLocation = {cameraSample.x,cameraSample.y,-depthForUnitSizePixel};
		pixelLocation.x = cameraSample.x - filmWidth / 2.f;
		pixelLocation.y = filmHeight/2.f - cameraSample.y;

		float3 origin = {0,0,0};
		Ray ray;

		ray.origin = eye;

		glm::vec4 rayDirectionCameraSpace = glm::vec4(to_vec3(pixelLocation - origin),1);
		glm::vec4 rayDirection = lookAtInv * rayDirectionCameraSpace;
		ray.direction = normalize(to_float3(rayDirection) / rayDirection.w);
		return ray;

	}
};
#pragma once

#include "../Core/Camera.h"
#include "../Utils/MathsCommons.h"
#include "../Core/Parameters.h"

class PerspectiveCamera: public Camera{
public:

	float fovX;
	float fovY;
	int filmWidth;
	int filmHeight;

	// depth value (z value) of a hypothetical film,
	// placed at the position which makes the pixels have size 1x1 in world space.
	float depthForUnitSizePixel;

	// turns a vec in camera space to the corresponding vec in world space
	glm::mat4 cameraToWorld;

	__host__ 
	PerspectiveCamera();

	__host__
	PerspectiveCamera(const glm::mat4& cameraToWorld , float fov_, int filmWidth_, int filmHeight_);

	
	static PerspectiveCamera createFromParams(const Parameters& params, const glm::mat4& cameraToWorld, int filmWidth_, int filmHeight_);


	__host__ __device__
	Ray genRay(const CameraSample& cameraSample) const
	{
		float3 pixelLocation = {cameraSample.x,cameraSample.y,depthForUnitSizePixel};
		pixelLocation.x = cameraSample.x - filmWidth / 2.f;
		pixelLocation.y = filmHeight/2.f - cameraSample.y;

		float3 origin = {0,0,0};
		Ray ray;

		glm::vec4 rayOriginCameraSpace = glm::vec4(to_vec3(origin),1);
		glm::vec4 rayOrigin = cameraToWorld * rayOriginCameraSpace;
		ray.origin = to_float3(rayOrigin) / rayOrigin.w;

		glm::vec4 rayDirectionCameraSpace = glm::vec4(to_vec3(pixelLocation - origin),1);
		glm::vec4 rayDirection = cameraToWorld * rayDirectionCameraSpace;
		ray.direction = normalize(to_float3(rayDirection) / rayDirection.w);

		return ray;

	}
};
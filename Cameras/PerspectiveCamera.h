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

		ray.origin = apply(cameraToWorld, origin);
		ray.direction = normalize(apply(cameraToWorld,pixelLocation) - ray.origin);


		return ray;

	}

	__host__ __device__
    virtual void pdf(const Ray &ray, float* outputPositionProbability, float* outputDirectionProbability) const override{

		float3 front = apply(cameraToWorld,make_float3(0,0,1));
		float cosTheta = dot(ray.direction, front);
		if (cosTheta <= 0) {
			*outputPositionProbability = 0;
			*outputDirectionProbability = 0;
			return;
		}
		
		*outputPositionProbability = 1;
		*outputDirectionProbability = 1.f / (cosTheta * cosTheta * cosTheta);
	}
};
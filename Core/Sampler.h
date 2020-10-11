#pragma once
#include "CameraSample.h"
#include "Camera.h"
#include "../Cameras/CameraObject.h"
#include "Film.h"
#include <vector>

class Sampler{
	public:

	virtual float rand1() = 0;

	virtual float2 rand2() = 0;
};

class CameraSampler {
public:
	virtual std::vector<CameraSample> genAllSamples(const CameraObject& camera, const Film& film)=0;
};
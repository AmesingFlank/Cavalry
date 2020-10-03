#pragma once
#include "CameraSample.h"
#include "Camera.h"
#include "Film.h"
#include <vector>

class Sampler{
	public:
	virtual float rand1() = 0;
	virtual float2 rand2() = 0;
	virtual float3 rand3() = 0;
	virtual float4 rand4() = 0;
};

class CameraSampler {
public:
	virtual std::vector<CameraSample> genAllSamples(const Camera& camera, const Film& film)=0;
};
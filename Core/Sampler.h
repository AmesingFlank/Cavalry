#pragma once
#include "CameraSample.h"
#include "Camera.h"
#include "Film.h"
#include <vector>

class CameraSampler {
public:
	virtual std::vector<CameraSample> genAllSamples(const Camera& camera, const Film& film)=0;
};
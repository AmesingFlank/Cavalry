#pragma once
#include "CameraSample.h"
#include "Camera.h"
#include "../Cameras/CameraObject.h"
#include "../Films/FilmObject.h"
#include "Film.h"
#include <vector>

#include <thrust/copy.h>
#include <thrust/device_vector.h>

class Sampler{
	public:

	__host__ __device__
	virtual float rand1() = 0;

	__host__ __device__
	virtual float2 rand2() = 0;
};

class CameraSampler {
public:
	virtual std::vector<CameraSample> genAllSamplesCPU(const CameraObject& camera, FilmObject& film)=0;
	virtual thrust::device_vector<CameraSample> genAllSamplesGPU(const CameraObject& camera, FilmObject& film)=0;
};


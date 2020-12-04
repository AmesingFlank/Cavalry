#pragma once
#include "CameraSample.h"
#include "Camera.h"
#include "../Cameras/CameraObject.h"
#include "../Films/FilmObject.h"
#include "Film.h"
#include <vector>
#include "../Utils/Array.h"
#include <thrust/copy.h>
#include <thrust/device_vector.h>

class Sampler{
public:

	int samplesPerPixel;

	virtual void prepare(int threadsCount) {};

	__device__
	virtual void startPixel() {};

	__device__
	virtual int randInt(int N) = 0;

	__device__
	virtual float rand1() = 0;

	__device__
	virtual float2 rand2() = 0;

	__device__
	virtual float4 rand4() = 0;

	virtual GpuArray<CameraSample> genAllCameraSamples(const CameraObject& camera, FilmObject& film) = 0;

	virtual void reorderStates(GpuArray<int>& taskIndices){};
};


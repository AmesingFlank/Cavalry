#pragma once
#include "CameraSample.h"
#include "Camera.h"
#include "../Cameras/CameraObject.h"
#include "Film.h"
#include <vector>
#include "../Utils/Array.h"
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include "SamplingState.h"

enum class CameraSamplingOrder {
	PixelByPixel,
	sppBySpp
};

class SamplerObject;

class Sampler{
public:

	int samplesPerPixel;

	int completedPixels = 0;
	int completedSPPs = 0;
	CameraSamplingOrder cameraSamplingOrder = CameraSamplingOrder::sppBySpp;

	virtual void prepare(int threadsCount) {};

	__device__
	virtual int randInt(int N,SamplingState& samplingState) = 0;

	__device__
	virtual float rand1(SamplingState& samplingState) = 0;

	__device__
	virtual float2 rand2(SamplingState& samplingState) = 0;

	__device__
	virtual float4 rand4(SamplingState& samplingState) = 0;

	virtual GpuArray<CameraSample> genAllCameraSamples(const CameraObject& camera, Film& film, int bytesNeededPerSample, int maxSamplesPerRound = -1);

	virtual SamplerObject getObjectFromThis() = 0;

	virtual int bytesNeededPerThread() = 0;
};


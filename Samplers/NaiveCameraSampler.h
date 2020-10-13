#pragma once

#include "../Core/Sampler.h"

class NaiveCameraSampler :public CameraSampler {
public:
	virtual std::vector<CameraSample> genAllSamplesCPU(const CameraObject& camera, FilmObject& film) override;
	virtual thrust::device_vector<CameraSample> genAllSamplesGPU(const CameraObject& camera, FilmObject& film) override;
};
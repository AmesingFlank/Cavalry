#pragma once

#include "RenderResult.h"
#include "Color.h"
#include "CameraSample.h"
#include "../Utils/GpuCommons.h"
#include "../Utils/Array.h"
#include "../Core/Parameters.h"


class Film {
public:

	Film();

	Film(int width_, int height_, bool isCopyForKernel_ = false);

	Film getCopyForKernel();

	int width;
	int height;
	int completedSamplesPerPixel = 0;


	__device__
	virtual void addSample(const CameraSample& sample, const Spectrum& spectrum) {


		int x0 = floor(sample.x);
		int x1 = ceil(sample.x);
		int y0 = floor(sample.y);
		int y1 = ceil(sample.y);
		for (int x = x0; x <= x1; ++x) {
			for (int y = y0; y <= y1; ++y) {
				if (!(x >= width || y >= height || x < 0 || y < 0)) {
					int index = y * width + x;
					atomicAdd(&(samplesSum.data[index]), spectrum);
					atomicAdd(&(samplesCount.data[index]), 1);
				}
			}
		}

	}

	virtual RenderResult readCurrentResult() ;

    GpuArray<Spectrum> samplesSum;
    GpuArray<int> samplesCount;

	static Film createFromParams(const Parameters& params);

};
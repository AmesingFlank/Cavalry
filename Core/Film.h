#pragma once

#include "RenderResult.h"
#include "Color.h"
#include "CameraSample.h"
#include "../Utils/GpuCommons.h"
#include "../Utils/Array.h"
#include "../Core/Parameters.h"
#include "../Filters/FilterObject.h"

class Film {
public:

	Film();

	Film(int width_, int height_, const FilterObject& filter_, bool isCopyForKernel_ = false);

	Film getCopyForKernel();

	int width;
	int height;
	int completedPixels = 0;

	FilterObject filter;

	__device__
	virtual void addSample(const CameraSample& sample, const Spectrum& spectrum) {
		if (!isfinite(spectrum.x) || !isfinite(spectrum.y) || !isfinite(spectrum.z)) {
			return;
		}

		int x0 = floor(sample.x-filter.xwidth());
		int x1 = ceil(sample.x + filter.xwidth());
		int y0 = floor(sample.y - filter.ywidth());
		int y1 = ceil(sample.y + filter.ywidth());

		for (int x = x0; x <= x1; ++x) {
			for (int y = y0; y <= y1; ++y) {
				if (!(x >= width || y >= height || x < 0 || y < 0)) {
					int index = y * width + x;
					float weight = filter.contribution(x, y, sample);
					if (weight > 0) {
						atomicAdd(&(samplesSum.data[index]), spectrum * weight);
						atomicAdd(&(samplesWeightSum.data[index]), weight);
					}
				}
			}
		}

	}

	virtual RenderResult readCurrentResult() ;

    GpuArray<Spectrum> samplesSum;
    GpuArray<float> samplesWeightSum;

	static Film createFromParams(const Parameters& params, const FilterObject& filter);

};
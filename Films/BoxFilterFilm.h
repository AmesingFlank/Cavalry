#pragma once

#include "../Core/Film.h"
#include "../Utils/GpuCommons.h"
#include "../Utils/Array.h"
#include "../Core/Parameters.h"


class BoxFilterFilm : public Film {
public:

	BoxFilterFilm();

	BoxFilterFilm(int width_, int height_, bool isCopyForKernel_ = false);

	BoxFilterFilm getCopyForKernel();


	__device__
	virtual void addSample(const CameraSample& sample, const Spectrum& spectrum) override{

        int x = round(sample.x-0.5);
		int y = round(sample.y-0.5);
		int index = y*width + x;

		if (x >= width || y >= height || x < 0 || y < 0) {
			printf("error %f %f %d %d %d %d\n",sample.x,sample.y, x, y, width, height);
		}
		
        atomicAdd(&(samplesSum.data[index]),spectrum);
        atomicAdd(&(samplesCount.data[index]), 1);
	}

	virtual RenderResult readCurrentResult()  override;

    GpuArray<Spectrum> samplesSum;
    GpuArray<int> samplesCount;

	static BoxFilterFilm createFromParams(const Parameters& params);

};
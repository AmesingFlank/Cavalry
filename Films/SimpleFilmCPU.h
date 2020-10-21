#pragma once

#include "../Core/Film.h"
#include <iostream>

class SimpleFilmCPU : public Film {
public:

	SimpleFilmCPU();

	SimpleFilmCPU(int width_, int height_);

	SimpleFilmCPU getCopyForKernel();

	__host__ __device__
	virtual void addSample(const CameraSample& sample, const Spectrum& spectrum) override{
#ifdef __CUDA_ARCH__
    	SIGNAL_ERROR("Not Implemented on GPU");
#else
		int x = round(sample.x - 0.5);
		int y = round(sample.y -0.5);
		int index = y*width + x;

		if (!(index*3 < result.data.size() && index >= 0)) {
			std::cout <<"simple film cpu error:   "<< x << " " << y <<"  "<<index<<"    "<<result.data.size()<< std::endl;
		}

		writeColorAt(spectrum,&(result.data[index*3]));
#endif
	}

	virtual RenderResult readCurrentResult()  override;

	RenderResult result;
};
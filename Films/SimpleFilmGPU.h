#pragma once

#include "../Core/Film.h"
#include "../Utils/GpuCommons.h"
#include "../Utils/Array.h"
#include "../Core/Parameters.h"


class SimpleFilmGPU : public Film {
public:

	SimpleFilmGPU();

	SimpleFilmGPU(int width_, int height_, bool isCopyForKernel_ = false);

	SimpleFilmGPU getCopyForKernel();


	__device__
	virtual void addSample(const CameraSample& sample, const Spectrum& spectrum) override{

        int x = round(sample.x-0.5);
		int y = round(sample.y-0.5);
		int index = y*width + x;
		if (x >= width || y >= height || x<0 || y<0) {
			SIGNAL_ERROR("error sample location %d %d %d %d\n", x, y, width, height);
		}
		writeColorAt(clampBetween0And1(spectrum),&(data.data[index*3]));
    	
		
	}

	virtual RenderResult readCurrentResult()  override;

	GpuArray<unsigned char> data;

	static SimpleFilmGPU createFromParams(const Parameters& params);

};
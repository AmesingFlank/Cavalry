#pragma once

#include "../Core/Film.h"
#include "../Utils/GpuCommons.h"


class SimpleFilmGPU : public Film {
public:

	SimpleFilmGPU();

	SimpleFilmGPU(int width_, int height_, bool isCopyForKernel_ = false);

	SimpleFilmGPU getCopyForKernel();


	__host__ __device__
	virtual void addSample(const CameraSample& sample, const Spectrum& spectrum) override{
#ifdef __CUDA_ARCH__
        int x = round(sample.x*(width-1));
		int y = round(sample.y*(height-1));
		int index = y*width + x;
		if (x >= width || y >= height) {
			printf("error %d %d %d %d\n", x, y, width, height);
		}
		writeColorAt(spectrum,&(data.data[index*3]));
    	
#else
        SIGNAL_ERROR("Not Implemented on CPU");
#endif
	}

	virtual RenderResult readCurrentResult()  override;

	ManagedArray<unsigned char> data;

};
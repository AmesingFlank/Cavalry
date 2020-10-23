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


	__host__ __device__
	virtual void addSample(const CameraSample& sample, const Spectrum& spectrum) override{
#ifdef __CUDA_ARCH__
        int x = round(sample.x-0.5);
		int y = round(sample.y-0.5);
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

	GpuArray<unsigned char> data;

	static SimpleFilmGPU createFromParams(const Parameters& params);

};
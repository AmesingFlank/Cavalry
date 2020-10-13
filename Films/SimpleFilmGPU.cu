#include "SimpleFilmGPU.h"

#include <iostream>
#include "../Utils/GpuCommons.h"

SimpleFilmGPU::SimpleFilmGPU():data(0,true){}

SimpleFilmGPU::SimpleFilmGPU(int width_, int height_,bool isCopyForKernel_):data(width_*height_*3, isCopyForKernel_){
	width = width_;
    height = height_;
}



RenderResult SimpleFilmGPU::readCurrentResult(){
    RenderResult result(width,height);
    HANDLE_ERROR(cudaMemcpy(result.data.data(),data.data,3*width*height*sizeof(unsigned char),cudaMemcpyDeviceToHost));
    return result;
}


SimpleFilmGPU SimpleFilmGPU::getCopyForKernel() {
    SimpleFilmGPU copy(width,height,true);
    copy.data = data.getCopyForKernel();
    return copy;
}
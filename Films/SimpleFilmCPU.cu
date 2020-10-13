#include "SimpleFilmCPU.h"

#include <iostream>
#include "../Utils/GpuCommons.h"

SimpleFilmCPU::SimpleFilmCPU():result(0,0){}

SimpleFilmCPU::SimpleFilmCPU(int width_, int height_):result(width_,height_){
	width = width_;
	height = height_;
}

RenderResult SimpleFilmCPU::readCurrentResult(){
    return result;
}


SimpleFilmCPU SimpleFilmCPU::getCopyForKernel(){
	SIGNAL_ERROR("not implemented. shouldn't pass to kernel");
}
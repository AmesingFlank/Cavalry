#include "SimpleFilm.h"

#include <iostream>
#include "../Utils/GpuCommons.h"

SimpleFilm::SimpleFilm():data(0,true){}

SimpleFilm::SimpleFilm(int width_, int height_,bool isCopyForKernel_):data(width_*height_*3, isCopyForKernel_){
	width = width_;
    height = height_;
}



RenderResult SimpleFilm::readCurrentResult(){
    RenderResult result(width,height);
    HANDLE_ERROR(cudaMemcpy(result.data.data(),data.data,3*width*height*sizeof(unsigned char),cudaMemcpyDeviceToHost));
    return result;
}


SimpleFilm SimpleFilm::getCopyForKernel() {
    SimpleFilm copy(width,height,true);
    copy.data = data.getCopyForKernel();
    return copy;
}

SimpleFilm SimpleFilm::createFromParams(const Parameters& params){
	int width = params.getNum("xresolution");
	int height = params.getNum("yresolution");
	return SimpleFilm(width,height,false);
}
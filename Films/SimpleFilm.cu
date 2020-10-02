#include "SimpleFilm.h"

#include <iostream>

SimpleFilm::SimpleFilm(int width_, int height_):result(width_,height_){
	width = width_;
	height = height_;
}

RenderResult SimpleFilm::readCurrentResult(){
    return result;
}

void SimpleFilm::addSample(const CameraSample& sample, const Color& color){
    int x = round(sample.x*(width-1));
    int y = round(sample.y*(height-1));
    int index = y*width + x;

	if (!(index*3 < result.data.size() && index >= 0)) {
		std::cout << x << " " << y <<"  "<<index<<"    "<<result.data.size()<< std::endl;
	}

	writeColorAt(color,&(result.data[index*3]));
}
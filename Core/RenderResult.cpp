#include "RenderResult.h"
#include <svpng.inc>

RenderResult::RenderResult(int width_, int height_) : width(width_), height(height_)
{
    data = new unsigned char[width*height*3];
	memset(data, 0, width * height * 3);
}

RenderResult::~RenderResult(){
    delete[] data;
}

void RenderResult::saveToPNG(const std::string& fileName){
    FILE *fp = fopen(fileName.c_str(), "wb");
    svpng(fp, width,height,data, 0);
}
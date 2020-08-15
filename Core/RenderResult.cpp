#include "RenderResult.h"
#include <svpng.inc>

RenderResult::RenderResult(int width_, int height_) : width(width_), height(height_)
{
    data = new unsigned char[width*height*3];
}

RenderResult::~RenderResult(){
    delete[] data;
}

void RenderResult::saveToPNG(const std::string& fileName){
    FILE *fp = fopen("rgb.png", "wb");
    svpng(fp, width,height,data, 0);
}
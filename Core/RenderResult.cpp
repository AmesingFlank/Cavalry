#include "RenderResult.h"
#include <svpng.inc>

RenderResult::RenderResult(int width_, int height_) : width(width_), height(height_),data(width_ * height_ * 3)
{
}



void RenderResult::saveToPNG(const std::string& fileName){
    FILE *fp = fopen(fileName.c_str(), "wb");
    svpng(fp, width,height,data.data(), 0);
}
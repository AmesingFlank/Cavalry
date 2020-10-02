#pragma once


#include <string>
#include <vector>

class RenderResult
{
public:
    unsigned int width;
    unsigned int height;
    std::vector<unsigned char> data;


    RenderResult(int width_, int height_);
    

    void saveToPNG(const std::string& fileName);
};
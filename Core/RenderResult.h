#include <string>

class RenderResult
{
public:
    unsigned char *data;
    unsigned int width;
    unsigned int height;

    RenderResult(int width_, int height_);
    ~RenderResult();

    void saveToPNG(const std::string& fileName);
};
#include "Texture.h"
#include <iostream>
#include "../Utils/Utils.h"
#include "Color.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../Dependencies/include/stb_image.h"

Texture2D Texture2D::createFromObjectDefinition(const ObjectDefinition& def, const glm::mat4& transform, const std::filesystem::path& basePath) {
    if (def.params.hasString("filename")) {
        std::string pathString = def.params.getString("filename");
        std::filesystem::path relativePath(pathString);
        std::string filename = (basePath / relativePath).generic_string();

        std::string postfix = getFileNamePostfix(filename);
        bool shouldInvertGamma = postfix == "tga" || postfix == "png";

        return Texture2D::createTextureFromFile(filename,shouldInvertGamma);
    }
    uchar4 data = make_uchar4(0, 0, 0, 0);
    auto result = Texture2D(1, 1, &data);
    return result;
}

Texture2D Texture2D::createTextureFromFile(const std::string& filename,bool shouldInvertGamma) {
    int width;
    int height;
    uchar4* data = (uchar4*)stbi_load(filename.c_str(), &width, &height, 0, STBI_rgb_alpha);
    std::cout << "texture size " << width << "  " << height << std::endl;

    if (shouldInvertGamma) {
        for (int i = 0; i < width*height; ++i) {
            uchar4& pixel = data[i];
            pixel.x = 255.f * inverseGammaCorrect(pixel.x / 255.f);
            pixel.y = 255.f * inverseGammaCorrect(pixel.y / 255.f);
            pixel.z = 255.f * inverseGammaCorrect(pixel.z / 255.f);
        }
    }

    Texture2D result(data,width, height);
    free((void*)data);
    return result;
}
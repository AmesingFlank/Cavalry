#pragma once

#include "Renderer.h"
#include <string>


struct RenderSetup{
    Renderer renderer;
    Scene scene;
    std::string outputFileName;
};
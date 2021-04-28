#pragma once

#include "Renderer.h"
#include <string>
#include "../Utils/Utils.h"

struct RenderSetup{
    Renderer renderer;
    Scene scene;
    std::string outputFileName;

    std::string getOutputFileName() {
        std::string fileName = "RenderOutput.png";
        if (outputFileName.size() > 0) {
            if (getFileNamePostfix(outputFileName) == "png") {
                fileName = outputFileName;
            }
            else {
                fileName = outputFileName + ".png";
            }
        }
        return fileName;
    }

    void prepareForRender() {
        int spp = renderer.integrator->sampler->getSamplesPerPixel();
        scene.prepareForRender(spp);
    }
};
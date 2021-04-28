#include <iostream>
#include <cxxopts.hpp>

#include "Core/Renderer.h"
#include "Core/Parameters.h"
#include "SceneLoading/SceneLoading.h"
#include "Utils/Timer.h"


void run(const Parameters& renderParameters) {
    RenderSetup setup = readRenderSetup(renderParameters);

    Timer::getInstance().start("Rendering");
    Timer::getInstance().start("Preparation");
    setup.prepareForRender();
    Timer::getInstance().stop("Preparation");
    auto& result = setup.renderer.render(setup.scene);
    Timer::getInstance().stop("Rendering");
    result.saveToPNG(setup.getOutputFileName());
    Timer::getInstance().printStatistics();
}


int main(int argc, char** argv){
    cxxopts::Options options("Cavalry", "A GPU Ray Tracer");

    options.add_options()
        ("i,input", "Input file name", cxxopts::value<std::string>())
        ("o,output", "Override output file name", cxxopts::value<std::string>())
        ("spp", "Override samples per pixel", cxxopts::value<int>())
        ("integrator", "Override integrator", cxxopts::value<std::string>())
        ("h,help", "Print usage")
    ;
    options.allow_unrecognised_options();

    auto args = options.parse(argc, argv);

    if (args.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    Parameters params;

    if (!args.count("input")) {
        std::cout << "No Input File Found, Rendering a default scene" << std::endl;

        std::string path = "C:/Users/Dunfan/Code/VSIDE/Cavalry/";

        std::string scenePath;
        //scenePath = path+"TestScenes/cornellBox/test6.pbrt";
        //scenePath = path+"TestScenes/bathroom2/scene.pbrt";
        //scenePath = path+"TestScenes/bathroom/bathroom.pbrt";
        //scenePath = path+"TestScenes/living-room-3/scene.pbrt";
        scenePath = path + "TestScenes/staircase/scene.pbrt";
        //scenePath = path+"TestScenes/staircase2/scene.pbrt";
        //scenePath = path+"TestScenes/ganesha/ganesha.pbrt";
        //scenePath = path+"TestScenes/coffee/scene.pbrt";
        //scenePath = path + "TestScenes/veach-mis2/scene.pbrt";
        //scenePath = path + "TestScenes/veach-mis/mis.pbrt";
        //scenePath = path + "TestScenes/veach-bidir2/scene.pbrt";
        //scenePath = path + "TestScenes/dragon/f8-21b.pbrt";
        //scenePath = path+"TestScenes/dragon/f9-3.pbrt";

        params.strings["input"] = scenePath;
    }
    else {
        params.strings["input"] = args["input"].as<std::string>();
    }

    if (args.count("output")) {
        params.strings["output"] = args["output"].as<std::string>();
    }
    if (args.count("integrator")) {
        params.strings["integrator"] = args["integrator"].as<std::string>();
    }
    if (args.count("spp")) {
        params.nums["spp"] = args["spp"].as<int>();
    }


    run(params);

    return 0;

   

  
}
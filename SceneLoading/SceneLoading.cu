#include "SceneLoading.h"
#include "Lexing.h"
#include "Parsing.h"
#include "../Utils/Utils.h"
#include <filesystem>

RenderSetup readRenderSetup(const Parameters& params){
    std::filesystem::path filePath = std::filesystem::path(params.getString("input"));
    std::filesystem::path basePath = filePath.parent_path();
    TokenBuf tokens = runLexing(filePath);
    RenderSetup setup = runParsing(tokens,basePath,params);
    setup.scene.BVHOptimizationRounds = (int)params.getNum("BVHOptimizationRounds");
    return setup;
}
#include "SceneLoading.h"
#include "Lexing.h"
#include "Parsing.h"
#include "../Utils/Utils.h"
#include <filesystem>

RenderSetup readRenderSetup(const std::string& pbrtFilePath){
    std::filesystem::path filePath = std::filesystem::path(pbrtFilePath);
    std::filesystem::path basePath = filePath.parent_path();
    TokenBuf tokens = runLexing(filePath);
    return runParsing(tokens,basePath);
}
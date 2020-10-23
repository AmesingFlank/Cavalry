#include "SceneLoading.h"
#include "Lexing.h"
#include "Parsing.h"
#include "../Utils/Utils.h"
#include <filesystem>

RenderSetup readRenderSetup(const std::string& pbrtFilePath){
    std::string contents = readTextFile(pbrtFilePath);
    TokenBuf tokens = runLexing(contents);
    return runParsing(tokens,std::filesystem::path(pbrtFilePath).parent_path());
}
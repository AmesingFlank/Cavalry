#include "SceneLoading.h"
#include "Lexing.h"
#include "Parsing.h"
#include "../Utils/Utils.h"

Scene readScene(const std::string& pbrtFilePath){
    std::string contents = readTextFile(pbrtFilePath);
    TokenBuf tokens = runLexing(contents);
    return runParsing(tokens);
}
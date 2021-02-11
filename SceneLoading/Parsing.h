#pragma once

#include "Lexing.h"
#include "../Core/RenderSetup.h"
#include "../Core/Parameters.h"
#include <filesystem>



RenderSetup runParsing(TokenBuf tokens,const std::filesystem::path& basePath, const Parameters& overridenParams);
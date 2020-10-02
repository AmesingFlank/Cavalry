#pragma once

#include "BSDF.h"
#include <vector>
#include <memory>
class Material{
public:
    std::vector<std::shared_ptr<BSDF>> bsdfs;
};
#pragma once

#include "BSDF.h"
#include <vector>
class Material{
public:
    std::vector<BSDF> bsdfs;
};
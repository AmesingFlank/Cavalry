#pragma once

#include "../BSDFs/BSDFObject.h"
#include <vector>
#include <memory>
#include "Ray.h"
#include "Color.h"
#include "IntersectionResult.h"
#include "../Core/Material.h"
#include "../Core/Texture.h"

class MatteMaterial:public Material{
public:
    LambertianBSDF lambertian;

    MatteMaterial():lambertian(make_float3(100,0,0)){}

    MatteMaterial(float3 color):lambertian(color){}


    __device__
    virtual BSDFObject getBSDF(const IntersectionResult& intersection) const override {
        return lambertian;
    }
};
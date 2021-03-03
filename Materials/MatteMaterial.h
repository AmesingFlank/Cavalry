#pragma once

#include "../BSDFs/BSDFObject.h"
#include <vector>
#include <memory>
#include "Ray.h"
#include "IntersectionResult.h"
#include "../Core/Material.h"
#include "../Core/Texture.h"
#include "../Core/Parameters.h"

class MatteMaterial:public Material{
public:
    Spectrum color;

    bool hasTexture = false;
    Texture2D texture;

    MatteMaterial():color(make_float3(100,0,0)), texture(0, 0, true){}

    MatteMaterial(float3 color_):color(color_), texture(0, 0, true){}

    MatteMaterial(float3 color_,const Texture2D& texture_):color(color_), texture(texture_),hasTexture(true) {}


    __device__
    virtual BSDFObject getBSDF(const IntersectionResult& intersection) const override {
        Spectrum thisColor = color;
        if (hasTexture) {
            float4 texel = texture.readTexture(intersection.textureCoordinates);
            Spectrum sampledColor = to_float3(texel);
            thisColor *= sampledColor;
            //thisColor = make_float3(intersection.textureCoordinates.x, intersection.textureCoordinates.y, 0);
        }
        return LambertianBSDF(thisColor);
    }

    __device__
    virtual MaterialType getType() const  override {
        return MaterialType::Matte;
    };

    static MatteMaterial createFromParams(const Parameters& params, const std::unordered_map<std::string, Texture2D>& textures) {
        Spectrum color = make_float3(1, 1, 1);
        if (params.hasNumList("Kd")) {
            auto colorVec = params.getNumList("Kd");
            color = make_float3(colorVec[0], colorVec[1], colorVec[2]);
        }
        if (params.hasString("Kd")) {
            std::string textureName = params.getString("Kd");
            return MatteMaterial(color, textures.at(textureName));
        }
        return MatteMaterial(color);
    }
};
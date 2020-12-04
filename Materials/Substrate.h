#pragma once

#include "../BSDFs/BSDFObject.h"
#include <vector>
#include <memory>
#include "Ray.h"
#include "Color.h"
#include "IntersectionResult.h"
#include "../Core/Material.h"
#include "../Core/Texture.h"
#include "../Core/Parameters.h"

class SubstrateMaterial:public Material{
public:
    Spectrum diffuseColor;
    bool hasDiffuseTexture = false;
    Texture2D diffuseTexture;
    GGX distribution;

    Spectrum specularColor;

    SubstrateMaterial():diffuseColor(make_float3(100,0,0)),specularColor(make_float3(100,0,0)), diffuseTexture(0, 0, true){}

    SubstrateMaterial(Spectrum diffuseColor_,Spectrum specularColor_,float uRoughness_,float vRoughness_,bool remapRoughness_):
    diffuseColor(diffuseColor_),specularColor(specularColor_), diffuseTexture(0, 0, true),
    distribution(GGX::createFromRoughness(uRoughness_,vRoughness_,remapRoughness_))
    {

    }

    SubstrateMaterial(Spectrum diffuseColor_,const Texture2D& diffuseTexture_,Spectrum specularColor_,float uRoughness_,float vRoughness_,bool remapRoughness_):
    diffuseColor(diffuseColor_),specularColor(specularColor_), diffuseTexture(diffuseTexture_),hasDiffuseTexture(true),
    distribution(GGX::createFromRoughness(uRoughness_,vRoughness_,remapRoughness_))
    {

    }


    __device__
    virtual BSDFObject getBSDF(const IntersectionResult& intersection) const override {
        Spectrum diffuse = diffuseColor;
        if (hasDiffuseTexture) {
            float4 texel = diffuseTexture.readTexture(intersection.textureCoordinates);
            diffuse = to_float3(texel);
        }
        return FresnelBlendBSDF(diffuse,specularColor,distribution);
    }

    __device__
    virtual MaterialType getType() const  override {
        return MaterialType::Substrate;
    };


    static SubstrateMaterial createFromParams(const Parameters& params, const std::unordered_map<std::string, Texture2D>& textures) {
        Spectrum diffuse = make_float3(0.5,0.5,0.5);
        if (params.hasNumList("Kd")) {
            auto colorVec = params.getNumList("Kd");
            diffuse = make_float3(colorVec[0], colorVec[1], colorVec[2]);
        }
        Spectrum specular = make_float3(0.5,0.5,0.5);
        if (params.hasNumList("Ks")) {
            auto colorVec = params.getNumList("Ks");
            specular = make_float3(colorVec[0], colorVec[1], colorVec[2]);
        }
        float uRoughness = 0.1;
        float vRoughness = 0.1;
        bool remap = true;
        if (params.hasNum("uroughness")) {
            uRoughness = params.getNum("uroughness");
        }
        if (params.hasNum("vroughness")) {
            vRoughness = params.getNum("vroughness");
        }
        if (params.hasString("remaproughness")) {
            remap = params.getString("remaproughness") == "true";
        }

        if (params.hasString("Kd")) {
            std::string textureName = params.getString("Kd");
            return SubstrateMaterial(diffuse, textures.at(textureName),specular,uRoughness,vRoughness,remap);
        }
        return SubstrateMaterial(diffuse, specular,uRoughness,vRoughness,remap);
    }
};
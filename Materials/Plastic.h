#pragma once


#include "../BSDFs/BSDFObject.h"
#include <vector>
#include <memory>
#include "Ray.h"
#include "IntersectionResult.h"
#include "../Core/Material.h"
#include "../Core/Texture.h"
#include "../Core/Parameters.h"

class PlasticMaterial:public Material{
public:
    GGX distribution;
    Fresnel fresnel;

    Spectrum diffuseColor;
    bool hasDiffuseTexture = false;
    Texture2D diffuseTexture;

    Spectrum specularColor;

    PlasticMaterial():diffuseTexture(0, 0, true) {}

    PlasticMaterial(Fresnel fresnel_,float roughness_,bool remapRoughness_, const Spectrum& diffuseColor_, const Spectrum& specularColor_):
    distribution(GGX::createFromRoughness(roughness_,roughness_,remapRoughness_)),fresnel(fresnel_),
    specularColor(specularColor_),diffuseColor(diffuseColor_),diffuseTexture(0,0,true)
    {
        
    }

    PlasticMaterial(Fresnel fresnel_,float roughness_,bool remapRoughness_, const Texture2D& diffuseTexture_, const Spectrum& specularColor_):
    distribution(GGX::createFromRoughness(roughness_,roughness_,remapRoughness_)),fresnel(fresnel_),
    specularColor(specularColor_),diffuseTexture(diffuseTexture_),hasDiffuseTexture(true)
    {
        
    }
    
    __device__
    virtual BSDFObject getBSDF(const IntersectionResult& intersection) const override {
        MicrofacetBSDF specular = MicrofacetBSDF(specularColor,distribution,fresnel);

        Spectrum thisDiffuseColor = diffuseColor; 
        if (hasDiffuseTexture) {
            float4 texel = diffuseTexture.readTexture(intersection.textureCoordinates);
            thisDiffuseColor = to_float3(texel);
        }
        LambertianBSDF diffuse = LambertianBSDF(thisDiffuseColor);
        //return specular;
        return PlasticBSDF(specular, diffuse);
    }

    __device__
    virtual MaterialType getType() const  override {
        return MaterialType::Plastic;
    };


    static PlasticMaterial createFromParams(const Parameters& params, const std::unordered_map<std::string, Texture2D>& textures) {
        Fresnel fresnel = Fresnel::createFromIOR(make_float3(1.5,1.5,1.5));
        
        float roughness = 0.1;
        bool remap = true;
        if (params.hasNum("roughness")) {
            roughness = params.getNum("roughness");
        }
        if (params.hasString("remaproughness")) {
            remap = params.getString("remaproughness") == "true";
        }

        Spectrum diffuse = make_float3(0.25,0.25,0.25);
        if (params.hasNumList("Kd")) {
            auto colorVec = params.getNumList("Kd");
            diffuse = make_float3(colorVec[0], colorVec[1], colorVec[2]);
        }
        Spectrum specular = make_float3(0.25,0.25,0.25);
        if (params.hasNumList("Ks")) {
            auto colorVec = params.getNumList("Ks");
            specular = make_float3(colorVec[0], colorVec[1], colorVec[2]);
        }

        if (params.hasString("Kd")) {
            std::string textureName = params.getString("Kd");
            return PlasticMaterial(fresnel,roughness,remap,textures.at(textureName),specular);
        }
        return PlasticMaterial(fresnel,roughness,remap,diffuse,specular);
    }
};
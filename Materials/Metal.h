#pragma once

#include "../BSDFs/BSDFObject.h"
#include <vector>
#include <memory>
#include "../Core/Material.h"
#include "../Core/Texture.h"
#include "../Core/Parameters.h"

class MetalMaterial:public Material{
public:
    GGX distribution;
    Fresnel fresnel;

    MetalMaterial(){}

    MetalMaterial(Fresnel fresnel_,float uRoughness_,float vRoughness_,bool remapRoughness_):
    distribution(GGX::createFromRoughness(uRoughness_,vRoughness_,remapRoughness_)),
    fresnel(fresnel_)
    {
        
    }
    
    __device__
    virtual BSDFObject getBSDF(const IntersectionResult& intersection) const override {
        
        return MicrofacetBSDF(make_float3(1,1,1),distribution,fresnel);
    }

    __device__
    virtual MaterialType getType() const  override {
        return MaterialType::Metal;
    };


    static MetalMaterial createFromParams(const Parameters& params, const std::unordered_map<std::string, Texture2D>& textures) {
        Spectrum copperF0 = make_float3(0.955,0.638,0.538);
        Fresnel fresnel = Fresnel::createFromF0(copperF0);
        
        if (params.hasNumList("eta") && params.hasNumList("k")) {
            auto etaVec = params.getNumList("eta");
            Spectrum eta = make_float3(etaVec[0], etaVec[1], etaVec[2]);
            auto kVec = params.getNumList("k");
            Spectrum k = make_float3(kVec[0], kVec[1], kVec[2]);
            fresnel = Fresnel::createFromIOR(eta,k);
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
        if (params.hasNum("roughness")) {
            uRoughness = params.getNum("roughness");
            vRoughness = params.getNum("roughness");
        }
        if (params.hasString("remaproughness")) {
            remap = params.getString("remaproughness") == "true";
        }

        return MetalMaterial(fresnel,uRoughness,vRoughness,remap);
    }
};
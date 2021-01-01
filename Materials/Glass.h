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

class GlassMaterial:public Material{
public:
    
    GGX distribution;
    Fresnel fresnel;
    Spectrum reflection;
    Spectrum transmission;
    float IOR;
    bool perfectSpecular;

    GlassMaterial(){}

    GlassMaterial(Spectrum reflection_,Spectrum transmission_,float IOR_,Fresnel fresnel_,float uRoughness_,float vRoughness_,bool remapRoughness_):
    distribution(GGX::createFromRoughness(uRoughness_,vRoughness_,remapRoughness_)),
    fresnel(fresnel_),reflection(reflection_),transmission(transmission_),IOR(IOR_),
    perfectSpecular(uRoughness_==0 && vRoughness_==0)
    {
        
    }
    
    __device__
    virtual BSDFObject getBSDF(const IntersectionResult& intersection) const override {
        if(perfectSpecular){
            return SpecularBSDF(reflection,fresnel,true,transmission,1.f,IOR);
        }
        return MicrofacetBSDF(reflection,distribution,fresnel,true,transmission,1.f,IOR);
    }

    __device__
    virtual MaterialType getType() const  override {
        return MaterialType::Glass;
    };


    static GlassMaterial createFromParams(const Parameters& params, const std::unordered_map<std::string, Texture2D>& textures) {
        Spectrum reflection = make_float3(1,1,1);
        Spectrum transmission = make_float3(1,1,1);
        float IOR = 1.5;
        if(params.hasNumList("Kr")){
            auto kr = params.getNumList("Kr");
            reflection = make_float3(kr[0],kr[1],kr[2]);
        }
        if(params.hasNumList("Kt")){
            auto kt = params.getNumList("Kt");
            transmission = make_float3(kt[0], kt[1], kt[2]);
        }
        if(params.hasNum("eta")){
            IOR = params.getNum("eta");
        }
        else if (params.hasNum("index")) {
            IOR = params.getNum("index");
        }
        Fresnel fresnel = Fresnel::createFromIOR(make_float3(IOR,IOR,IOR));
        
        float uRoughness = 0.;
        float vRoughness = 0.;
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

        return GlassMaterial(reflection,transmission,IOR,fresnel,uRoughness,vRoughness,remap);
    }
};
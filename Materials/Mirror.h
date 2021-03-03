#pragma once


#include "../BSDFs/BSDFObject.h"
#include <vector>
#include <memory>
#include "../Core/Material.h"
#include "../Core/Texture.h"
#include "../Core/Parameters.h"

class MirrorMaterial:public Material{
public:
    
    float reflectivity;

    MirrorMaterial():reflectivity(0.9f){}

    MirrorMaterial(float reflectivity_):reflectivity(reflectivity_){}


    __device__
    virtual BSDFObject getBSDF(const IntersectionResult& intersection) const override {
        return MirrorBSDF(reflectivity);
    }

    __device__
    virtual MaterialType getType() const  override {
        return MaterialType::Mirror;
    };


    static MirrorMaterial createFromParams(const Parameters& params, const std::unordered_map<std::string, Texture2D>& textures) {
        
        if (params.hasNum("Kr")) {
            return MirrorMaterial(params.getNum("Kr"));
        }
        return MirrorMaterial();
    }
};
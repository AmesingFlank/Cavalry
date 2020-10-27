#include "PerspectiveCamera.h"

PerspectiveCamera::PerspectiveCamera(){}


PerspectiveCamera::PerspectiveCamera(const glm::mat4& cameraToWorld_, float fov_, int filmWidth_, int filmHeight_):cameraToWorld(cameraToWorld_),filmWidth(filmWidth_),filmHeight(filmHeight_){

    if(filmHeight <= filmWidth){
        fovY = fov_;
        float z = (filmHeight/2.f)/tan(fovY/2.f);
        fovX = 2.f * atan2(filmWidth/2,z);
        depthForUnitSizePixel = z;
    }
    else{
        fovX = fov_;
        float z = (filmWidth/2.f)/tan(fovX/2.f);
        fovY = 2.f * atan2(filmHeight/2,z);
        depthForUnitSizePixel = z;
    }
}

PerspectiveCamera  PerspectiveCamera::createFromParams(const Parameters& params, const glm::mat4& cameraToWorld , int filmWidth_, int filmHeight_)
{
    float fov = glm::radians(params.getNum("fov"));
    return PerspectiveCamera(cameraToWorld,fov,filmWidth_,filmHeight_);
}
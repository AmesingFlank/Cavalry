#include "PerspectiveCamera.h"

PerspectiveCamera::PerspectiveCamera(){}


PerspectiveCamera::PerspectiveCamera(const float3& eye_, const float3& center_, const float3& up_, float fov_, int filmWidth_, int filmHeight_):eye(eye_),center(center_),up(up_),filmWidth(filmWidth_),filmHeight(filmHeight_){
    lookAtInv = glm::inverse(glm::lookAt(to_vec3(eye),to_vec3(center),to_vec3(up)));

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
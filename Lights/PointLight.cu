#include "PointLight.h"

PointLight::PointLight(float3 position_, Spectrum color_):position(position_),color(color_){

}

Spectrum PointLight::sampleRayToPoint(const float3& position,const float2& randomSource, float& outputProbability, Ray& outputRay){
    outputProbability = 1;

    outputRay.origin = position;
    outputRay.direction = normalize(this->position - position);

    return color;
}
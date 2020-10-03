#include "Lambertian.h"
#include "../Utils/MathsCommons.h"



LambertianBSDF::LambertianBSDF(const Color& baseColor_):baseColor(baseColor_){

}

Color LambertianBSDF::eval(float3 incident, float3 exitant){
    return baseColor / M_PI;
}
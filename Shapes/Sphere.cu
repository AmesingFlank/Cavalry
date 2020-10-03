#include "Sphere.h"
#include <iostream>

Sphere::Sphere(float3 center_,float radius_):center(center_),radius(radius_){

}

bool Sphere::intersect(IntersectionResult& result, const Ray& ray){
    float3 oc = ray.origin - center;
    float a = 1;
    float b = 2 * dot(ray.direction, oc);
    float c = dot(oc,oc) - radius*radius;
    float discriminant = b*b - 4*a*c;
    if(discriminant < 0){
        result.intersected = false;
        //std::cout << "no intersection.. d<0" <<std::endl;
        return false;
    }
    float t1 = (-b - sqrt(discriminant)) / (2*a);
    float t2 = (-b + sqrt(discriminant)) / (2*a);

    if(t1 <= 0 && t2 <= 0){
        result.intersected = false;
        //std::cout << "no intersection.. t<0" <<std::endl;
        return false;
    }

    float t = t1;
    if(t1 <=0 ) {
        t = t2;
    }


    result.distance = t;
    result.position = ray.positionAtDistance(t);
    result.normal = normalize(result.position - center);
    result.intersected = true;
    //std::cout << "intersected" <<std::endl;
    return true;

}
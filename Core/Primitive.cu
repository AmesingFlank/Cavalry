#include "Primitive.h"

bool Primitive::intersect(IntersectionResult& result, const Ray& ray){
    if(!shape->intersect(result, ray)) return false;
    result.primitive = this;
    return true;
}
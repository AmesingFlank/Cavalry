#include "../Core/BSDF.h"

class LambertianBSDF: public BSDF{
public:
    Color baseColor;
    LambertianBSDF(const Color& baseColor_);
    virtual Color eval(float3 incident, float3 exitant) override;
};
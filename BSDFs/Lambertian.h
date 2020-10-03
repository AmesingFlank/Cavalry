#include "../Core/BSDF.h"

class LambertianBSDF: public BSDF{
public:
    Spectrum baseColor;
    LambertianBSDF(const Spectrum& baseColor_);
    virtual Spectrum eval(float3 incident, float3 exitant) override;
};
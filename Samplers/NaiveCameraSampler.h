#pragma

#include "../Core/Sampler.h"

class NaiveCameraSampler:public CameraSampler {
public:
    std::vector<CameraSample> genAllSamples(const Camera& camera, const Film& film) override;
}
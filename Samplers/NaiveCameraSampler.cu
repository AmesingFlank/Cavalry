#include "NaiveCameraSampler.h"

std::vector<CameraSample> NaiveCameraSampler::genAllSamples(const Camera& camera, const Film& film) {
    int width = film.width;
    int height = film.height;
    std::vector<CameraSample> result;
    for(float x = 0;x<width;x += 1){
        for(float y = 0;y<height; y += 1){
            CameraSample sample {x/(float) (width-1),y/(float)(height-1)};
            result.push_back(sample);
        }
    }
    return result;
}
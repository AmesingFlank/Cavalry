#include "SimpleSampler.h"
#include "../Utils/GpuCommons.h"
#include "DecideSampleCount.h"
#include "../Utils/RandomUtils.h"
#include "../Utils/Utils.h"
#include <iostream>
#include "SamplerObject.h"


SimpleSampler::SimpleSampler(int samplesPerPixel_, bool isCopyForKernel_ ):states(1024,isCopyForKernel_){
    samplesPerPixel = samplesPerPixel_;
}

SimpleSampler::SimpleSampler() :states(0,true) {

}

SimpleSampler SimpleSampler::getCopyForKernel(){
    SimpleSampler copy(samplesPerPixel,true);
    copy.states = states.getCopyForKernel();
    return copy;
}




SamplerObject SimpleSampler::getObjectFromThis() {
    return getCopyForKernel();
}
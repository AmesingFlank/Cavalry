#include "SimpleSamplerGPU.h"
#include "../Utils/GpuCommons.h"

#include "../Utils/RandomUtils.h"


SimpleSamplerGPU::SimpleSamplerGPU(int maxThreads_, bool isCopyForKernel_ ):maxThreads(maxThreads_),states(1024,isCopyForKernel_){
    
}

SimpleSamplerGPU::SimpleSamplerGPU() : maxThreads(-1), states(0,true) {

}

SimpleSamplerGPU SimpleSamplerGPU::getCopyForKernel(){
    SimpleSamplerGPU copy(maxThreads,true);
    copy.states = states.getCopyForKernel();
    return copy;
}
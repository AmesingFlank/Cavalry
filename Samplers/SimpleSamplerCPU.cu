#include "SimpleSamplerCPU.h"
#include "../Utils/GpuCommons.h"

#include "../Utils/RandomUtils.h"


SimpleSamplerCPU SimpleSamplerCPU::getCopyForKernel(){
    SIGNAL_ERROR("not implemented. shouldn't be passed to kernel")
}
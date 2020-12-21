#include "BVHNode.h"

void collapseNodes(int primitivesCount,GpuArray<BVHNode>& nodes);


void optimizeBVH(int primitivesCount,GpuArray<BVHNode>& nodes);

void genOptimizationSchedule();
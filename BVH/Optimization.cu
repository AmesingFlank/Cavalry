#include "../Utils/GpuCommons.h"
#include "Optimization.h"
#include "../Utils/Array.h"
#include "../Utils/Utils.h"
#include <vector>
#include <cooperative_groups.h>
using namespace cooperative_groups;

namespace cg = cooperative_groups;

using byte = unsigned char;

std::vector<unsigned int> scheduleCPU = {

/*round 1*/
0b00011,0b00110,0b00101,0b01001,0b01010,0b01100,0b10001,0b10010,0b10100,0b11000, /*2bits; greatest at 5 or less*/ 
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, /*round 1 vacancies*/

/*round 2*/
0b00111,0b01011,0b01101,0b01110,0b10011,0b10101,0b10110,0b11001,0b11010,0b11100,/*3bits; greatest at 5 or less*/     
0b100001,0b100010,0b100100,0b101000,0b110000,/*2 bits; 6th bit set*/  
0b1000001,0b1000010,0b1000100,0b1001000,0b1010000,0b1100000/*2 bits; 7th bit set*/, 
0,0,0,0,0,0,0,0,0,0,0, /*round 2 vacancies*/


/*round 3*/
0b100011,0b100101,0b100110,0b101001,0b101010,0b101100,0b110001,0b110010,0b110100,0b111000, /*3bits; 6th set*/
0b1000011,0b1000101,0b1000110,0b1001001,0b1001010,0b1001100,0b1010001,0b1010010,0b1010100,0b1011000,0b1100001,0b1100010,0b1100100,0b1101000,0b1110000,/*3bits; 7th set*/
0b0001111,0b0010111,0b0011110, /*some of 4bits*/
0,0,0,0, /*round 3 vacancies*/


/*round 4; all the remaining 4 bits*/
0b1000111,0b0011011,0b0101011,0b1001011,0b0110011,0b1010011,0b1100011,0b0011101,0b0101101,0b1001101,0b0110101,0b1010101,0b1100101,0b0111001,0b1011001,0b1101001,0b1110001,0b0100111,0b0101110,0b1001110,0b0110110,0b1010110,0b1100110,0b0111010,0b1011010,0b1101010,0b1110010,0b0111100,0b1011100,0b1101100,0b1110100,0b1111000,

/*round 5; all the 5 bits*/
0b0011111,0b0101111,0b1001111,0b0110111,0b1010111,0b1100111,0b0111011,0b1011011,0b1101011,0b1110011,0b0111101,0b1011101,0b1101101,0b1110101,0b1111001,0b0111110,0b1011110,0b1101110,0b1110110,0b1111010,0b1111100,
0,0,0,0,0,0,0,0,0,0,0

};

__device__ 
void optimizeTreelet(BVHRestructureNode* nodes, int root, thread_block_tile<32> thisWarp, int warpIndex, float* area, float* optimalCost, byte* optimalPartition) {
    int laneIndex = thisWarp.thread_rank();// ==index % 32;

    int myNode = -1;
    if (laneIndex == 0) myNode = root;

    // Treelet formation
    int currentTreeletSize = 1;
    bool expandedMe = false;
    while (currentTreeletSize < 2 * 7 - 1) {

        int nodeToExpand;

        if (myNode != -1) {
            float myArea = nodes[myNode].surfaceArea;

            if (nodes[myNode].isLeaf || expandedMe) myArea = -1; // if myNode is a leaf, it shouldn't be considered for expansion.

            int maxLane = 0;
            float maxArea = -1;
            for (int i = 0; i < currentTreeletSize; ++i) {
                float thatArea = thisWarp.shfl(myArea, i);
                if (thatArea > maxArea) {
                    maxArea = thatArea;
                    maxLane = i;
                }
            }

            if (maxLane == laneIndex) {
                expandedMe = true;
                nodeToExpand = myNode;
            }
            nodeToExpand = thisWarp.shfl(nodeToExpand, maxLane);
        }

        nodeToExpand = thisWarp.shfl(nodeToExpand, 0);

        if (laneIndex == currentTreeletSize) {
            myNode = nodes[nodeToExpand].leftChild;
        }
        if (laneIndex == currentTreeletSize + 1) {
            myNode = nodes[nodeToExpand].rightChild;
        }
        currentTreeletSize += 2;
    }
    int currentLeafID = 0;
    int leaves[7] = { -1,-1,-1,-1,-1,-1,-1 };

    for (int i = 0; i < 2 * 7 - 1; ++i) {
        if (thisWarp.shfl((int)(!expandedMe), i)) {
            int leafNode = thisWarp.shfl(myNode, i);
            leaves[currentLeafID] = leafNode;
            ++currentLeafID;
        }
    }

    if (currentLeafID != 7 && laneIndex == 0) {
        SIGNAL_ERROR("wrong! should have 7 leaves\n");
    }
    // treelet formation is now done;

    // compute AABB area of all subsets
    AABB mySubsetBox;
    bool hasInitialBox = false;
    for (int bitPos = 0; bitPos < 5; ++bitPos) {
        bool thisLeafIsSelected = 1 & (laneIndex >> bitPos);
        if (thisLeafIsSelected) {
            if (!hasInitialBox) {
                mySubsetBox = nodes[leaves[bitPos]].box;
                hasInitialBox = true;
            }
            else {
                mySubsetBox = unionBoxes(mySubsetBox, nodes[leaves[bitPos]].box);
            }
        }
    }
    for (int i = 0; i < 4; ++i) {
        byte subset = (i << 5) | laneIndex;
        AABB thisBox = mySubsetBox;
        bool thisBoxIsNonEmpty = hasInitialBox;
        for (int bitPos = 5; bitPos < 7; ++bitPos) {
            bool thisLeafIsSelected = 1 & (subset >> bitPos);
            if (thisLeafIsSelected) {
                if (!thisBoxIsNonEmpty) {
                    thisBox = nodes[leaves[bitPos]].box;
                    thisBoxIsNonEmpty = true;
                }
                else {
                    thisBox = unionBoxes(thisBox, nodes[leaves[bitPos]].box);
                }
            }
        }
        if (thisBoxIsNonEmpty) {
            area[subset] = thisBox.computeSurfaceArea();
        }
        else {
            area[subset] = 0;
        }
    }

    float areaError = area[127] - nodes[root].surfaceArea;
    if (abs(areaError) > 1e-3 && laneIndex==0) {
        SIGNAL_ERROR("wrong! big area error\n");
    }
    // done computing areas for eachsubset


    // Initialize costs of individual leaves
    for (int i = 0; i < 7; ++i) {
        byte singleton = 1 << i;

    }

}

//1152 = 4*128 (for area) + 4*128(for cost) + 1*128(for partition)
#define BYTES_NEEDED_PER_WARP 1152  

__global__
void optimizeBVHImpl(int nodesCount, BVHRestructureNode* nodes, unsigned int* visited){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= nodesCount*32) return;

    thread_block_tile<32> thisWarp = tiled_partition<32>(this_thread_block());

    int warpIndex = index / 32;
    int laneIndex = thisWarp.thread_rank();// ==index % 32;

    int curr = warpIndex;

    if (!nodes[curr].isLeaf) return;
    // skip over bottom 3 levels, so that there're at least 7 leaves
    for (int i = 0; i < 3; ++i) {
        int parent = nodes[curr].parent;
        bool getParent = false;
        if (laneIndex == 0) {
            getParent = atomicInc(&visited[parent], 2) == 1;
        }
        getParent = thisWarp.shfl(getParent, 0);
        if (!getParent) return;
        curr = parent;
    }

    extern __shared__ byte sharedMem[];
    int warpIndexInBlock = threadIdx.x / 32;
    byte* thisWarpSharedMem = sharedMem + warpIndexInBlock * BYTES_NEEDED_PER_WARP;
    float* area = (float*)thisWarpSharedMem;
    float* optimalCost = (float*)(thisWarpSharedMem + 128 * 4);
    byte* optimalPartition = (byte*)(thisWarpSharedMem + 128 * 4 + 128*4);

    while (true) {
        optimizeTreelet(nodes, curr, thisWarp,warpIndex,area,optimalCost,optimalPartition);
        if (curr == 0) break;

        int parent = nodes[curr].parent;
        bool getParent = false;
        if (laneIndex == 0) {
            getParent = atomicInc(&visited[parent], 2) == 1;
        }
        getParent = thisWarp.shfl(getParent, 0);
        if (!getParent) return;
        curr = parent;
    }
    
}

void optimizeBVH(int primitivesCount,GpuArray<BVHRestructureNode>& nodes){
    int nodesCount = nodes.N;
    int threadsNeeded = nodesCount * 32;

    int numBlocks,numThreads;
    setNumBlocksThreads(threadsNeeded,numBlocks,numThreads);

    GpuArray<unsigned int> visited(nodesCount,false);

    optimizeBVHImpl <<< numBlocks,numThreads, BYTES_NEEDED_PER_WARP * numThreads / 32 >>> (nodesCount,nodes.data,visited.data);
    CHECK_IF_CUDA_ERROR("optimize bvh");
}
#include "../Utils/GpuCommons.h"
#include "Optimization.h"
#include "SAH.h"
#include "../Utils/Array.h"
#include "../Utils/Utils.h"
#include <vector>
#include <cooperative_groups.h>


// this file implements the paper: Fast Parallel Construction of High-Quality Bounding Volume Hierarchies, by karras and aila


using namespace cooperative_groups;

namespace cg = cooperative_groups;

// we consider treelets of 7 leaves; Each subset of leaves is represented using a byte;
using byte = char;

__device__ byte optimizationSchedule[160] = {

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
void optimizeTreelet(BVHRestructureNode *  nodes, int root, thread_block_tile<32> thisWarp, int warpIndex, float* area, float* optimalCost, byte* optimalPartition) {
    int laneIndex = thisWarp.thread_rank();// ==index % 32;

    int myNode = -1;
    if (laneIndex == 0) {
        myNode = root;
    }

    // Treelet formation. Repeated expand the node with the greatest area, and assign it to two vacant threads.
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
                if (expandedMe || nodes[myNode].isLeaf) {
                    // expandedMe is already true, this means that there aren't enough nodes in this Treelet;
                    // so we set nodeToExpand to -1, broadcast this information to all nodes in the warp, and return;
                    nodeToExpand = -1;
                }
                else {
                    expandedMe = true;
                    nodeToExpand = myNode;
                }
            }
            nodeToExpand = thisWarp.shfl(nodeToExpand, maxLane);
        }

        // broardCast nodeToExpand to nodes who didn't participate in its selection
        nodeToExpand = thisWarp.shfl(nodeToExpand, 0);
        if (nodeToExpand == -1) {
            return;
        }

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
        SIGNAL_ERROR("error! should have 7 leaves\n");
    }
    
    // treelet formation is now done;

    // compute AABB area of all subsets. There're 128 subsets, each thread handles 4 of them.
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
        optimalCost[subset] = -1; // initialize cost as -1 for all;
    }

    float areaError = area[127] - nodes[root].surfaceArea;
    if (abs(areaError) > 1e-3 && laneIndex==0) {
        SIGNAL_ERROR("error! big area error %f at root %d\n",areaError,root);
    }
    // done computing areas for eachsubset
    
    // Initialize costs of individual leaves
    for (int i = 0; i < 7; ++i) {
        byte singleton = 1 << i;
        optimalCost[singleton] = nodes[leaves[i]].cost;
    }

    // Optimize every subset of leaves of size 2 to 5 (inclusive), using a pre-defined hard-coded schedule.
    // The schedule ensures that the values needed in each round are fully available in previosu rounds; 
    // The schedule assigns a different subset for each thread in the warp, and the thread enumerates all partitions
    for (int round = 0; round < 5; ++round) {
        byte subset = optimizationSchedule[round * 32 + laneIndex];
        if (subset != 0) {

            float bestCost = -1;
            byte bestPartition = 0;

            // the following lines iterates through all possible partitions of the subset
            // bit operation insanity; check paper for details
            byte subsetClearedFirstBit = (subset - 1) & subset;

            byte partition = (0 - subsetClearedFirstBit) & subset;
            do {
                float partitionCost = optimalCost[partition];

                byte remaining = subset ^ partition;
                float remainingCost = optimalCost[remaining];

                float thisTotalCost = partitionCost + remainingCost;
                if (partitionCost < 0) {
                    SIGNAL_ERROR("error! cost = %f < 0,  %d\n", partitionCost,(int)partition);
                }
                if (remainingCost < 0) {
                    SIGNAL_ERROR("error! cost = %f < 0,  %d\n", remainingCost, (int)remaining);
                }
                if (bestCost == -1 || thisTotalCost < bestCost) {
                    bestPartition = partition;
                    bestCost = thisTotalCost;
                }
                partition = (partition - subsetClearedFirstBit) & subset;

            } while (partition != 0);

            optimalCost[subset] = internalCost(area[subset],bestCost);
            optimalPartition[subset] = bestPartition;
        }
    }


// a small helper function
#define consequtiveBits(n) ((1<<n)-1)

    // there're 7 subsets with 6 bits, and each one of these has 31 partitions
    // each thread works on a single partition, and the results are collected via parallel reduction;
    for (int emptyBit = 0; emptyBit < 7; ++emptyBit) {
        byte subset = 0b1111111 ^ (1 << emptyBit);

        // partition always has the highest bit unset
        // to understand how this works, use an example: consider when emptyBit = 3, and thus subset = 1110111, 
        byte partition =
            (laneIndex & consequtiveBits(emptyBit))
            |
            ((laneIndex << 1) & ( consequtiveBits(5-emptyBit) << (emptyBit + 1)))
            ;
        
        
        float totalCost = -1;
        if (partition != 0) {
            byte remaining = subset ^ partition;
            float partitionCost = optimalCost[partition];
            float remainingCost = optimalCost[remaining];

            if (partitionCost < 0) {
                SIGNAL_ERROR("in 6bits, error! cost = %f < 0,  %d\n", partitionCost, (int)partition);
            }
            if (remainingCost < 0) {
                SIGNAL_ERROR("in 6bits, error! cost = %f < 0,  %d\n", remainingCost, (int)remaining);
            }
            totalCost = partitionCost + remainingCost;
        }

        //parallel min reduction
        for (int d = 16; d >= 1; d = d>>1) {
            int otherLane = min(laneIndex+d, 31);
            float otherCost = thisWarp.shfl(totalCost, otherLane);
            byte otherPartition = (byte)(thisWarp.shfl((int)partition, otherLane));
            
            if (totalCost == -1 || otherCost < totalCost) {
                totalCost = otherCost;
                partition = otherPartition;
            }
        }
        if (laneIndex == 0) {
            optimalCost[subset] = internalCost(area[subset], totalCost);
            optimalPartition[subset] = partition;
        }
    }


    // there're 1 subset with 7 bits, and it has 63 partitions
    // each thread works on 2 partitions, and the results are collected via parallel reduction;
    {
        byte subset = 0b1111111;
        byte partition = laneIndex;
        float totalCost = -1;

        if (partition != 0) {
            byte remaining = subset ^ partition;
            float partitionCost = optimalCost[partition];
            float remainingCost = optimalCost[remaining];

            if (partitionCost < 0) {
                SIGNAL_ERROR("in 7bits, error! cost = %f < 0,  %d\n", partitionCost, (int)partition);
            }
            if (remainingCost < 0) {
                SIGNAL_ERROR("in 7bits, error! cost = %f < 0,  %d\n", remainingCost, (int)remaining);
            }
            totalCost = partitionCost + remainingCost;
        }
        

        {
            byte otherPartition = laneIndex | (1 << 5);
            byte otherRemaining = subset ^ otherPartition;
            float otherPartitionCost = optimalCost[otherPartition];
            float otherRemainingCost = optimalCost[otherRemaining];

            if (otherPartitionCost < 0) {
                SIGNAL_ERROR("in 7bits, error! cost = %f < 0,  %d\n", otherPartitionCost, (int)otherPartition);
            }
            if (otherRemainingCost < 0) {
                SIGNAL_ERROR("in 7bits, error! cost = %f < 0,  %d\n", otherRemainingCost, (int)otherRemaining);
            }
            float otherTotalCost = otherPartitionCost + otherRemainingCost;
            if (totalCost == -1 || otherTotalCost < totalCost) {
                totalCost = otherTotalCost;
                partition = otherPartition;
            }
        }

        //parallel min reduction
        for (int d = 16; d >= 1; d = d >> 1) {
            int otherLane = min(laneIndex+d, 31);
            float otherCost = thisWarp.shfl(totalCost, otherLane);
            byte otherPartition = (byte)(thisWarp.shfl((int)partition, otherLane));
            if (totalCost == -1 || otherCost < totalCost) {
                totalCost = otherCost;
                partition = otherPartition;
            }
        }
        if (laneIndex == 0) {
            optimalCost[subset] = internalCost(area[subset], totalCost);
            optimalPartition[subset] = partition;
        }
    }
    // optimal partitions found
    

    // reconstruct tree. Re-use the original nodes
    
    byte mySubset = -1;
    if (laneIndex == 0) {
        mySubset = 0b1111111;
    }
    int reconstructedSize = 1;
    bool expandedInReconstruction = false;
    int myParent = -1;
    if (laneIndex == 0) {
        myParent = nodes[myNode].parent;
    }

    if (myNode != -1) {
        nodes[myNode].newLeftChild = -1;
        nodes[myNode].newRightChild = -1;
    }
    
    
    while (reconstructedSize < 2 * 7 - 1) {
        byte subsetToExpand = -1;
        int subsetToExpandLane = -1;
        if (mySubset != -1) {
            byte myProposal = -1;

#define isPowerOf2(n) ((n & (n-1)) == 0)

            if (!expandedInReconstruction && !isPowerOf2(mySubset)) {
                myProposal = mySubset;
            }
            //printf("%d proposing %d\n", laneIndex, myProposal);
            for (int i = 0; i < reconstructedSize; ++i) {
                byte proposal = (byte)thisWarp.shfl((int)myProposal, i);
                if (proposal != -1 && subsetToExpand == -1) {
                    subsetToExpand = proposal;
                    subsetToExpandLane = i;
                }
            }

            if (subsetToExpand <= 0 || subsetToExpand > 127 || isPowerOf2(subsetToExpand)) {
                //if (subsetToExpandLane == laneIndex)
                SIGNAL_ERROR("error! subsetToExpand is %d,  %d %d\n",(int)subsetToExpand, (int)mySubset, (int)myProposal);
            }

            if (subsetToExpand == mySubset) {
                expandedInReconstruction = true;
            }
        }

        // broadcast to nodes who didn't participate in selecting subsetToExpand. src=0 because 0th thread always has the result.
        subsetToExpand = thisWarp.shfl(subsetToExpand, 0);
        subsetToExpandLane = thisWarp.shfl(subsetToExpandLane, 0);

        int parentNode = thisWarp.shfl(myNode, subsetToExpandLane);

        if (laneIndex == reconstructedSize) {
            myParent = parentNode;
            mySubset = optimalPartition[subsetToExpand];
            nodes[myParent].newLeftChild = myNode;
        }
        if (laneIndex == reconstructedSize + 1) {
            myParent = parentNode;
            mySubset = subsetToExpand ^ optimalPartition[subsetToExpand];
            nodes[myParent].newRightChild = myNode;
        }
        reconstructedSize += 2;
    }
    bool newIsLeaf;
    int newPrimBegin;
    int newPrimEnd;
    AABB newBox;

    if (myNode != -1) {
        nodes[myNode].parent = myParent;
        if (!expandedInReconstruction) {
            int leafID = 0;
            byte temp = mySubset;
            while ((temp & 1) == 0) {
                ++leafID;
                temp = temp >> 1;
            }
            int originalLeaf = leaves[leafID];
            newIsLeaf = nodes[originalLeaf].isLeaf;
            newPrimBegin = nodes[originalLeaf].primitiveIndexBegin;
            newPrimEnd = nodes[originalLeaf].primitiveIndexEnd;
            newBox = nodes[originalLeaf].box;

            nodes[myNode].newLeftChild = nodes[originalLeaf].leftChild;
            nodes[myNode].newRightChild = nodes[originalLeaf].rightChild;

            if (!nodes[originalLeaf].isLeaf) {
                // if the treelet leaf is an internal node, then we need to point its children to the correct new parent
                nodes[nodes[originalLeaf].leftChild].parent = myNode;
                nodes[nodes[originalLeaf].rightChild].parent = myNode;
            }
        }
        else {
            newIsLeaf = false;
        }
    }
    if (myNode != -1) {
        nodes[myNode].leftChild = nodes[myNode].newLeftChild;
        nodes[myNode].rightChild = nodes[myNode].newRightChild;
        nodes[myNode].isLeaf = newIsLeaf;
        nodes[myNode].primitiveIndexBegin = newPrimBegin;
        nodes[myNode].primitiveIndexEnd = newPrimEnd;
        nodes[myNode].box = newBox;
        nodes[myNode].surfaceArea = newBox.computeSurfaceArea();

        
        if (nodes[myNode].isLeaf) {
            //printf("for %d, is leaf, prims [%d,%d], parent is %d\n",myNode, nodes[myNode].primitiveIndexBegin, nodes[myNode].primitiveIndexEnd,nodes[myNode].parent);
        }
        else {
            //printf("for %d, left child is %d, right child is %d, parent is %d\n", myNode, nodes[myNode].leftChild, nodes[myNode].rightChild, nodes[myNode].parent);
        }

        for (int i = 0; i < 7; ++i) {
            if (expandedInReconstruction) {
                auto& leftChild = nodes[nodes[myNode].leftChild];
                auto& rightChild = nodes[nodes[myNode].rightChild];
                nodes[myNode].box = unionBoxes(leftChild.box, rightChild.box);
                nodes[myNode].surfaceArea = nodes[myNode].box.computeSurfaceArea();
            }
            if (nodes[myNode].isLeaf) {
                nodes[myNode].cost = leafCost(nodes[myNode].surfaceArea, nodes[myNode].primitivesCount());
            }
            else {
                auto& leftChild = nodes[nodes[myNode].leftChild];
                auto& rightChild = nodes[nodes[myNode].rightChild];
                nodes[myNode].cost = internalCost(nodes[myNode].surfaceArea, leftChild.cost, rightChild.cost);
            }
        }        
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

    extern __shared__ byte sharedMem[];
    int warpIndexInBlock = threadIdx.x / 32;
    byte* thisWarpSharedMem = sharedMem + warpIndexInBlock * BYTES_NEEDED_PER_WARP;
    float* area = (float*)thisWarpSharedMem;
    float* optimalCost = (float*)(thisWarpSharedMem + 128 * 4);
    byte* optimalPartition = (byte*)(thisWarpSharedMem + 128 * 4 + 128*4);

    while (true) {
        //if (laneIndex == 0) printf("visiting %d\n", curr);
        if(laneIndex==0 && curr == 0) printf("original BVH cost %f\n", nodes[0].cost);
        optimizeTreelet(nodes, curr, thisWarp,warpIndex,area,optimalCost,optimalPartition);
        __threadfence();
        if (laneIndex == 0 && curr == 0) printf("optimized BVH cost %f\n", nodes[0].cost);

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

void optimizeBVH(int primitivesCount,GpuArray<BVHRestructureNode>& nodes, int optimizationRounds){
    int nodesCount = nodes.N;
    int threadsNeeded = nodesCount * 32;

    int numBlocks,numThreads;
    setNumBlocksThreads(threadsNeeded,numBlocks,numThreads);

    GpuArray<unsigned int> visited(nodesCount,false);

    for (int i = 0; i < optimizationRounds; ++i) {
        visited.clear();

        optimizeBVHImpl << < numBlocks, numThreads, BYTES_NEEDED_PER_WARP* numThreads / 32 >> > (nodesCount, nodes.data, visited.data);
        CHECK_IF_CUDA_ERROR("optimize bvh");
    }
    
}

